"""
finetune_gpt4o_mini.py — Fine-tune GPT-4o-mini via OpenAI API for News Agent (L2).

OPTIONAL for the ensemble. After completion, set OPENAI_FT_NEWS_MODEL in .env.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI
from fine_tuning.generate_training_data import generate_gpt_finetune_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JSONL_PATH = Path("fine_tuning/data/gpt_finetune_train.jsonl")
RESULTS_PATH = Path("fine_tuning/data/gpt_ft_result.json")
BASE_MODEL = "gpt-4o-mini-2024-07-18"
SUFFIX = "supply-chain-news-agent"


def run_gpt_finetuning() -> str:
    """
    End-to-end GPT-4o-mini fine-tuning via OpenAI API for the News Agent.

    Generates JSONL training data, uploads to OpenAI, creates a fine-tuning job,
    polls until completion, and saves the resulting model ID to gpt_ft_result.json.
    Requires OPENAI_API_KEY. Training runs on OpenAI servers (no local GPU).

    Returns:
        Fine-tuned model ID string (set as OPENAI_FT_NEWS_MODEL in .env).
    """
    # Step 1 — Authenticate with OpenAI. Requires OPENAI_API_KEY in .env.
    # Unlike DistilBERT/embeddings, training runs on OpenAI servers (no local GPU).
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    client = OpenAI(api_key=api_key)

    # Step 2 — Build JSONL training file if not present. Each row is a
    # {messages: [system, user, assistant]} conversation: system prompt from
    # news_agent.py, user = disruption context fields, assistant = ground-truth
    # JSON (category, severity, regions, commodities, duration, summary).
    # Sourced from semiconductor_signals rows with known disruption events.
    if not JSONL_PATH.exists():
        generate_gpt_finetune_jsonl()

    # Step 3 — Upload JSONL to OpenAI Files API (purpose=fine-tune).
    # Returns a file_id referenced by the fine-tuning job.
    with open(JSONL_PATH, "rb") as f:
        upload_resp = client.files.create(file=f, purpose="fine-tune")
    file_id = upload_resp.id
    logger.info("Uploaded training file: %s", file_id)

    # Step 4 — Create a fine-tuning job on gpt-4o-mini. OpenAI handles
    # hyperparameter selection (batch_size, LR multiplier) automatically.
    # suffix becomes part of the resulting model name for easy identification.
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=BASE_MODEL,
        suffix=SUFFIX,
        hyperparameters={"n_epochs": 3, "batch_size": "auto", "learning_rate_multiplier": "auto"},
    )
    job_id = job.id
    logger.info("Fine-tuning job created: %s", job_id)

    # Step 5 — Poll job status every 30s until succeeded/failed/cancelled.
    # Typical runtime: 10–30 min depending on example count and OpenAI queue.
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        logger.info("Status: %s", job.status)
        if job.status == "succeeded":
            model_id = job.fine_tuned_model
            logger.info("Fine-tuning complete: %s", model_id)
            break
        elif job.status in ("failed", "cancelled"):
            raise RuntimeError(f"Job {job_id} failed: {job.error}")
        time.sleep(30)

    # Step 6 — Save job metadata locally and print .env line for News Agent L2.
    # Set OPENAI_FT_NEWS_MODEL=<model_id> so news_agent.py uses the fine-tuned
    # model instead of base gpt-4o-mini for structured JSON extraction.
    result = {"file_id": file_id, "job_id": job_id, "model_id": model_id}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Add to .env: OPENAI_FT_NEWS_MODEL=%s", model_id)
    return model_id


if __name__ == "__main__":
    run_gpt_finetuning()
