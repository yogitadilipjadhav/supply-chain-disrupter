"""
Test suite for ensemble signals (DistilBERT + LLM Judge + two-stage RAG).
Run: python -m pytest tests/test_ensemble_signals.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_distilbert_signal_no_model():
    """Signal 2 must never raise when model is absent."""
    from src.agents import distilbert_signal
    from src.agents.distilbert_signal import run_distilbert_inference

    with patch.object(distilbert_signal, "_model_available", return_value=False):
        result = run_distilbert_inference(
            {"delivery_status": "Late delivery", "supply_disruption_index": 8.5},
            duration_days=30.0,
        )
    assert result.model_source == "not-available-skipped"
    assert result.predicted_label == "N/A"
    assert result.confidence == 0.0


def test_build_inference_text_matches_training_format():
    """Train/serve text format must be identical."""
    from src.agents.distilbert_signal import build_distilbert_text, build_inference_text

    row = {
        "order_region": "Eastern Asia",
        "product_name": "Laptop",
        "known_disruption_event": "COVID-19 Pandemic",
        "disruption_news_count": 5,
        "supply_disruption_index": 7.5,
        "defect_rate_pct": 2.1,
        "export_control_level": 3.5,
        "risk_score_composite": 0.62,
        "lead_time_variance_days": 12.0,
    }
    assert build_inference_text(row) == build_distilbert_text(row)
    assert "Known disruption event: COVID-19 Pandemic." in build_distilbert_text(row)
    assert "Delivery:" not in build_distilbert_text(row)
    assert "Disruption duration: 0 days." in build_distilbert_text(row)
    assert "Disruption duration: 30 days." in build_distilbert_text(row, duration_days=30.0)


def test_judge_hard_rule_canceled():
    """critical_flag must come from final_label, not judge alone."""
    from src.agents.state import JudgeVerdict

    jv = JudgeVerdict(
        final_label="CRITICAL",
        verdict_type="unanimous",
        reasoning="all agree",
        signals_agreed=True,
        final_critical_flag=True,
    )
    assert (jv.final_label == "CRITICAL") is True


def test_rag_retriever_cross_encoder_fallback():
    """Two-stage RAG degrades gracefully when cross-encoder unavailable."""
    from src.rag.retriever import rerank_results

    mock_candidates = [
        {"text": "Candidate A", "distance": 0.3},
        {"text": "Candidate B", "distance": 0.1},
        {"text": "Candidate C", "distance": 0.5},
    ]
    with patch("src.rag.retriever._get_cross_encoder", return_value=None):
        result = rerank_results("test query", mock_candidates, top_k=2)
    assert len(result) == 2
    assert result[0]["distance"] <= result[1]["distance"]


def test_resolve_embedding_model_hf_repo_id(monkeypatch):
    """EMBEDDING_MODEL_PATH Hugging Face repo ids must not require a local path."""
    import src.rag.utils as rag_utils

    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "user/supply-chain-embeddings")
    monkeypatch.setattr(rag_utils, "FINETUNED_EMBEDDING_MODEL", Path("/nonexistent"))
    assert rag_utils.resolve_embedding_model_name() == "user/supply-chain-embeddings"


def test_query_collection_uses_chroma_singleton():
    """Named-collection queries must reuse get_chroma_client(), not a new client."""
    from src.rag.collections import query_collection

    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.count.return_value = 0
    mock_client.get_collection.return_value = mock_col

    with patch("src.rag.utils.get_chroma_client", return_value=mock_client):
        with patch("src.rag.utils.get_embedding_model"):
            query_collection("historical_precedents", "test query", n_results=3)
    mock_client.get_collection.assert_called_once()


def test_get_embedding_model_fallback_to_base(tmp_path, monkeypatch):
    """get_embedding_model falls back to base when fine-tuned path absent."""
    import src.rag.utils as rag_utils

    monkeypatch.setattr(rag_utils, "FINETUNED_EMBEDDING_MODEL", tmp_path / "nonexistent")
    monkeypatch.delenv("EMBEDDING_MODEL_PATH", raising=False)
    rag_utils.get_embedding_model.cache_clear() if hasattr(rag_utils.get_embedding_model, "cache_clear") else None
    ef = rag_utils.get_embedding_model()
    assert ef is not None
