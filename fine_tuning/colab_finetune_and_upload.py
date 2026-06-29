"""
colab_finetune_and_upload.py — One-shot GPU fine-tuning for Google Colab + Hugging Face upload.

Run both DistilBERT (4-class risk) and sentence-transformer (RAG bi-encoder) fine-tuning,
then push models to Hugging Face Hub for use in the local project.

══════════════════════════════════════════════════════════════════════════════
GOOGLE COLAB QUICKSTART
══════════════════════════════════════════════════════════════════════════════

1. Runtime → Change runtime type → **T4 GPU** (or better).

2. Upload your project folder (zip) and unzip, OR clone from GitHub:
     !git clone https://github.com/YOUR_USER/supply-chain-disrupter.git
     %cd supply-chain-disrupter

3. Ensure the Excel file exists:
     data/raw/supply_chain_lite_master.xlsx

4. Create a Hugging Face token (write access):
     https://huggingface.co/settings/tokens

5. Set secrets below (or use Colab Secrets: HF_TOKEN, HF_USERNAME).

6. Run:
     !python fine_tuning/colab_finetune_and_upload.py

7. After training, add to your local `.env`:
     DISTILBERT_MODEL_ID=your-username/supply-chain-distilbert-risk
     EMBEDDING_MODEL_PATH=your-username/supply-chain-embeddings

8. Rebuild ChromaDB locally (embeddings changed):
     python scripts/build_rag_collections.py --flush

══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# ── USER CONFIG (edit before running in Colab) ────────────────────────────────
HF_USERNAME = os.getenv("HF_USERNAME", "your-hf-username")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # or set in Colab: os.environ["HF_TOKEN"] = "hf_..."

DISTILBERT_REPO = os.getenv("DISTILBERT_REPO", f"{HF_USERNAME}/supply-chain-distilbert-risk")
EMBEDDING_REPO = os.getenv("EMBEDDING_REPO", f"{HF_USERNAME}/supply-chain-embeddings")

SKIP_DB_BUILD = os.getenv("SKIP_DB_BUILD", "0") == "1"
SKIP_DISTILBERT = os.getenv("SKIP_DISTILBERT", "0") == "1"
SKIP_EMBEDDINGS = os.getenv("SKIP_EMBEDDINGS", "0") == "1"
SKIP_HF_UPLOAD = os.getenv("SKIP_HF_UPLOAD", "0") == "1"
MAKE_DOWNLOAD_ZIP = os.getenv("MAKE_DOWNLOAD_ZIP", "1") == "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCEL_PATH = PROJECT_ROOT / "data" / "raw" / "supply_chain_lite_master.xlsx"
SQLITE_PATH = PROJECT_ROOT / "outputs" / "supply_chain.db"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def install_dependencies() -> None:
    """Install training deps (Colab usually needs this once per session)."""
    packages = [
        "transformers>=4.40.0",
        "torch",
        "datasets>=2.18.0",
        "accelerate>=0.27.0",
        "sentence-transformers>=5,<6",
        "scikit-learn>=1.3.0",
        "huggingface_hub>=0.23.0",
        "openpyxl>=3.1,<4",
        "pandas>=2.2,<4",
        "chromadb>=1.5,<2",
        "pypdf>=6,<7",
        "python-docx>=1.2,<2",
        "pyyaml>=6,<7",
    ]
    logger.info("Installing / verifying dependencies...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
    )


def login_huggingface() -> None:
    if SKIP_HF_UPLOAD:
        logger.info("SKIP_HF_UPLOAD=1 — skipping Hugging Face login.")
        return
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is required for upload. Set os.environ['HF_TOKEN'] = 'hf_...' "
            "or export HF_TOKEN before running."
        )
    from huggingface_hub import login

    login(token=HF_TOKEN)
    logger.info("Logged in to Hugging Face as %s", HF_USERNAME)


def ensure_database() -> None:
    """Build SQLite from Excel. ChromaDB is rebuilt after embedding fine-tune."""
    if SKIP_DB_BUILD and SQLITE_PATH.exists():
        logger.info("SKIP_DB_BUILD=1 and DB exists — skipping build.")
        return
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Excel not found: {EXCEL_PATH}\n"
            "Upload data/raw/supply_chain_lite_master.xlsx before running."
        )
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    logger.info("Building SQLite from %s ...", EXCEL_PATH.name)
    from src.utils.etl_loader import get_sqlite_stats, load_excel_into_sqlite

    inserted = load_excel_into_sqlite(flush_existing=True)
    logger.info("SQLite: loaded %d Lite Master rows", inserted)
    logger.info("%s", json.dumps(get_sqlite_stats(), indent=2))
    logger.info(
        "ChromaDB build deferred until after embedding fine-tune "
        "(avoids loading weights from an empty fine_tuning/models/ dir)."
    )


def rebuild_chromadb() -> None:
    """Rebuild ChromaDB using the fine-tuned (or base) embedding model."""
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.rag.utils import build_rag_corpus_complete

    logger.info("=== Rebuilding ChromaDB with fine-tuned embeddings ===")
    summary = build_rag_corpus_complete(flush_existing=True)
    logger.info("ChromaDB:\n%s", json.dumps(summary, indent=2))


def generate_training_data() -> None:
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    from fine_tuning.generate_training_data import load_distilbert_data, save_all_qa_pairs

    logger.info("=== Generating DistilBERT training split ===")
    load_distilbert_data()
    logger.info("=== Generating RAG QA pairs ===")
    save_all_qa_pairs()


def train_distilbert() -> Path:
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

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

    from fine_tuning.generate_training_data import ID2LABEL, LABEL2ID, load_distilbert_data

    model_output = PROJECT_ROOT / "fine_tuning" / "models" / "distilbert_risk_classifier"
    data_dir = PROJECT_ROOT / "fine_tuning" / "data"
    max_length = 256

    logger.info("=== Fine-tuning DistilBERT (GPU=%s) ===", torch.cuda.is_available())
    X_train, y_train, X_val, y_val = load_distilbert_data()

    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    val_ds = Dataset.from_dict({"text": X_val, "label": y_val})
    dataset = DatasetDict({"train": train_ds, "validation": val_ds})

    base_model = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenised = dataset.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_length, padding=False),
        batched=True,
        remove_columns=["text"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(model_output / "checkpoints"),
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
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
        report_to="none",
        seed=42,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
            "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    model_output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_output))
    tokenizer.save_pretrained(str(model_output))

    preds = trainer.predict(tokenised["validation"])
    y_pred = np.argmax(preds.predictions, axis=-1)
    report = classification_report(
        y_val, y_pred, target_names=[ID2LABEL[i] for i in range(4)], digits=4
    )
    f1_macro = float(f1_score(y_val, y_pred, average="macro"))
    logger.info("DistilBERT validation F1 macro: %.4f\n%s", f1_macro, report)

    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "distilbert_val_metrics.json", "w") as f:
        json.dump({"f1_macro": f1_macro, "per_class_report": report}, f, indent=2)

    return model_output


def train_embeddings() -> Path:
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    import random

    from sentence_transformers import InputExample, SentenceTransformer, losses
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from torch.utils.data import DataLoader

    from fine_tuning.finetune_embeddings import _accuracy_at_3
    from fine_tuning.generate_training_data import save_all_qa_pairs

    model_output = PROJECT_ROOT / "fine_tuning" / "models" / "supply_chain_embeddings"
    data_path = PROJECT_ROOT / "fine_tuning" / "data" / "qa_pairs.json"

    logger.info("=== Fine-tuning RAG embeddings ===")
    if not data_path.exists():
        save_all_qa_pairs()

    with open(data_path) as f:
        pairs = json.load(f)
    logger.info("Loaded %d QA pairs.", len(pairs))

    random.seed(42)
    eval_pairs = random.sample(pairs, min(100, len(pairs) // 5))
    eval_queries = {f"q{i}": p["query"] for i, p in enumerate(eval_pairs)}
    corpus = {f"c{i}": p["positive"] for i, p in enumerate(eval_pairs)}
    relevant_docs = {f"q{i}": {f"c{i}"} for i in range(len(eval_pairs))}
    eval_query_texts = set(eval_queries.values())
    train_pairs = [p for p in pairs if p["query"] not in eval_query_texts]

    train_examples = [InputExample(texts=[p["query"], p["positive"]]) for p in train_pairs]
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=32)

    base_model = "all-MiniLM-L6-v2"
    model = SentenceTransformer(base_model)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    evaluator = InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="supply_chain_ir",
        show_progress_bar=True,
    )

    model_output.mkdir(parents=True, exist_ok=True)
    baseline_result = evaluator(model, output_path=str(model_output / "eval"))
    baseline_score = _accuracy_at_3(baseline_result)
    logger.info("Embedding baseline Accuracy@3: %.4f", baseline_score)

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=3,
        warmup_steps=100,
        optimizer_params={"lr": 2e-5},
        evaluator=evaluator,
        evaluation_steps=200,
        output_path=str(model_output),
        save_best_model=True,
        show_progress_bar=True,
    )

    finetuned = SentenceTransformer(str(model_output))
    final_result = evaluator(finetuned)
    final_score = _accuracy_at_3(final_result)
    improvement = final_score - baseline_score
    logger.info(
        "Embedding fine-tuned Accuracy@3: %.4f (baseline %.4f, +%.1f%%)",
        final_score,
        baseline_score,
        improvement * 100,
    )

    with open(model_output / "retrieval_metrics.json", "w") as f:
        json.dump(
            {
                "baseline_accuracy_at_3": baseline_score,
                "finetuned_accuracy_at_3": final_score,
                "improvement": improvement,
                "baseline_full": baseline_result if isinstance(baseline_result, dict) else {},
                "finetuned_full": final_result if isinstance(final_result, dict) else {},
            },
            f,
            indent=2,
        )
    return model_output


def upload_to_huggingface(distilbert_path: Path, embedding_path: Path) -> None:
    if SKIP_HF_UPLOAD:
        logger.info("Skipping Hugging Face upload.")
        return

    from huggingface_hub import HfApi
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    api = HfApi()
    for repo_id in (DISTILBERT_REPO, EMBEDDING_REPO):
        api.create_repo(repo_id, exist_ok=True, repo_type="model")
        logger.info("Repo ready: https://huggingface.co/%s", repo_id)

    logger.info("Uploading DistilBERT → %s", DISTILBERT_REPO)
    tokenizer = AutoTokenizer.from_pretrained(str(distilbert_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(distilbert_path))
    model.push_to_hub(DISTILBERT_REPO, commit_message="Fine-tuned 4-class supply chain risk classifier")
    tokenizer.push_to_hub(DISTILBERT_REPO, commit_message="Tokenizer for supply chain DistilBERT")

    card_distilbert = f"""---
language: en
license: apache-2.0
tags:
  - distilbert
  - text-classification
  - supply-chain
  - risk-classification
datasets:
  - custom
---

# Supply Chain DistilBERT Risk Classifier

Fine-tuned `distilbert-base-uncased` for 4-class electronics supply chain risk:
**LOW | MEDIUM | HIGH | CRITICAL**

Trained on `lite_master` rows from the supply-chain-disrupter project.

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_id = "{DISTILBERT_REPO}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "Region: Eastern Asia. Product: DRAM. Delivery: Late delivery. ..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
with torch.no_grad():
    pred = model(**inputs).logits.argmax(-1).item()
labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
print(labels[pred])
```

## Local project

Set in `.env`:
```
DISTILBERT_MODEL_ID={DISTILBERT_REPO}
```
"""
    api.upload_file(
        path_or_fileobj=card_distilbert.encode(),
        path_in_repo="README.md",
        repo_id=DISTILBERT_REPO,
        repo_type="model",
        commit_message="Add model card",
    )

    logger.info("Uploading embeddings → %s", EMBEDDING_REPO)
    from sentence_transformers import SentenceTransformer

    emb_model = SentenceTransformer(str(embedding_path))
    emb_model.push_to_hub(
        EMBEDDING_REPO,
        commit_message="Fine-tuned supply chain RAG bi-encoder (MultipleNegativesRankingLoss)",
    )

    card_embeddings = f"""---
language: en
license: apache-2.0
tags:
  - sentence-transformers
  - feature-extraction
  - supply-chain
  - rag
---

# Supply Chain RAG Embeddings

Fine-tuned `all-MiniLM-L6-v2` for supply-chain RAG retrieval (historical precedents,
export controls, India sourcing, mitigation QA pairs).

## Usage

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("{EMBEDDING_REPO}")
q = model.encode("Red Sea shipping disruption semiconductor")
```

## Local project

Set in `.env`:
```
EMBEDDING_MODEL_PATH={EMBEDDING_REPO}
```

Then rebuild ChromaDB:
```bash
python scripts/build_rag_collections.py --flush
```
"""
    api.upload_file(
        path_or_fileobj=card_embeddings.encode(),
        path_in_repo="README.md",
        repo_id=EMBEDDING_REPO,
        repo_type="model",
        commit_message="Add model card",
    )

    logger.info("Upload complete.")
    logger.info("  DistilBERT:  https://huggingface.co/%s", DISTILBERT_REPO)
    logger.info("  Embeddings:  https://huggingface.co/%s", EMBEDDING_REPO)


def make_models_zip(distilbert_path: Path, embedding_path: Path) -> Path | None:
    if not MAKE_DOWNLOAD_ZIP:
        return None
    zip_path = PROJECT_ROOT / "fine_tuning" / "models" / "finetuned_models_for_local.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder, arc_prefix in [
            (distilbert_path, "distilbert_risk_classifier"),
            (embedding_path, "supply_chain_embeddings"),
        ]:
            for file in folder.rglob("*"):
                if file.is_file():
                    zf.write(file, arcname=str(Path(arc_prefix) / file.relative_to(folder)))
    logger.info("Saved local backup zip: %s", zip_path)
    if _in_colab():
        try:
            from google.colab import files

            files.download(str(zip_path))
            logger.info("Triggered Colab download of finetuned_models_for_local.zip")
        except Exception as exc:
            logger.warning("Could not auto-download zip: %s", exc)
    return zip_path


def print_local_setup(distilbert_path: Path, embedding_path: Path) -> None:
    print("\n" + "=" * 72)
    print("DONE — use models locally")
    print("=" * 72)
    print("\nOption A — Hugging Face (recommended, no large files in git):")
    print(f"  DISTILBERT_MODEL_ID={DISTILBERT_REPO}")
    print(f"  EMBEDDING_MODEL_PATH={EMBEDDING_REPO}")
    print("\nOption B — copy folders from zip / Colab download:")
    print(f"  {distilbert_path}")
    print(f"  {embedding_path}")
    print("\nAfter embeddings, rebuild ChromaDB:")
    print("  python scripts/build_rag_collections.py --flush")
    print("=" * 72 + "\n")


def main() -> None:
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Colab: %s | CUDA requested via torch after install", _in_colab())

    install_dependencies()
    login_huggingface()
    ensure_database()
    generate_training_data()

    distilbert_path = PROJECT_ROOT / "fine_tuning" / "models" / "distilbert_risk_classifier"
    embedding_path = PROJECT_ROOT / "fine_tuning" / "models" / "supply_chain_embeddings"

    if not SKIP_DISTILBERT:
        distilbert_path = train_distilbert()
    elif not distilbert_path.exists():
        raise FileNotFoundError("SKIP_DISTILBERT=1 but no saved DistilBERT model found.")

    if not SKIP_EMBEDDINGS:
        embedding_path = train_embeddings()
    elif not embedding_path.exists():
        raise FileNotFoundError("SKIP_EMBEDDINGS=1 but no saved embedding model found.")

    from src.rag.utils import _embedding_weights_present

    if _embedding_weights_present(embedding_path):
        rebuild_chromadb()
    else:
        logger.warning("Embedding weights not found — skipping ChromaDB rebuild.")

    upload_to_huggingface(distilbert_path, embedding_path)
    make_models_zip(distilbert_path, embedding_path)
    print_local_setup(distilbert_path, embedding_path)


if __name__ == "__main__":
    main()
