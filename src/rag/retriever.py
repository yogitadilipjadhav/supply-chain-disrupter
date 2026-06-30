"""
retriever.py — Two-stage RAG retrieval for Capstone Project 8.

Stage 1: Fine-tuned (or base) all-MiniLM bi-encoder retrieves top-N candidates
         per ChromaDB collection using cosine distance.
Stage 2: Cross-encoder (ms-marco-MiniLM-L-6-v2) reranks candidates by reading
         (query, document) pairs jointly with full attention.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from src.rag.collections import query_collection

logger = logging.getLogger(__name__)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Load cross-encoder once and cache. Returns None if unavailable."""
    try:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder: %s", CROSS_ENCODER_MODEL)
        t0 = time.monotonic()
        model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
        logger.info("Cross-encoder loaded in %.2fs", time.monotonic() - t0)
        return model
    except ImportError:
        logger.warning(
            "sentence-transformers CrossEncoder not available — reranking disabled."
        )
        return None
    except Exception as exc:
        logger.warning("Cross-encoder load failed: %s", exc)
        return None


def rerank_results(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Stage 2: Rerank bi-encoder candidates using cross-encoder joint scoring.

    Falls back to bi-encoder distance sort when cross-encoder is unavailable.
    """
    if not candidates:
        return []

    model = _get_cross_encoder()
    if model is None:
        logger.warning("Cross-encoder not available — returning bi-encoder top-%d", top_k)
        return sorted(candidates, key=lambda x: x.get("distance", 1.0))[:top_k]

    t0 = time.monotonic()
    pairs = [(query, hit["text"]) for hit in candidates]
    scores = model.predict(pairs)

    for hit, score in zip(candidates, scores):
        hit["cross_encoder_score"] = float(score)
        hit["bi_encoder_distance"] = hit.get("distance", 1.0)

    reranked = sorted(
        candidates, key=lambda x: x["cross_encoder_score"], reverse=True
    )[:top_k]
    logger.info(
        "Cross-encoder: %d → %d in %.0fms  top-score=%.3f",
        len(candidates),
        len(reranked),
        (time.monotonic() - t0) * 1000,
        reranked[0]["cross_encoder_score"] if reranked else 0.0,
    )
    return reranked


def retrieve_and_rerank(
    query: str,
    collections: List[str],
    bi_encoder_top_n: int = 10,
    rerank_top_k: int = 3,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Full two-stage retrieval across multiple named ChromaDB collections.

    Stage 1 queries each collection; Stage 2 pools and reranks all candidates.
    """
    all_candidates: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for cname in collections:
        hits = query_collection(
            collection_name=cname,
            query_text=query,
            n_results=bi_encoder_top_n,
            where=where,
        )
        for hit in hits:
            meta = hit.get("metadata", {})
            dedup_key = meta.get("source_file", meta.get("source", "")) + hit["text"][:40]
            if dedup_key not in seen_ids:
                seen_ids.add(dedup_key)
                all_candidates.append(hit)

    logger.info(
        "Stage 1 (bi-encoder): %d candidates from %d collections",
        len(all_candidates),
        len(collections),
    )
    return rerank_results(query, all_candidates, top_k=rerank_top_k)


def format_rag_chunks(
    chunks: List[Dict[str, Any]],
    query_label: str = "",
    include_scores: bool = True,
) -> str:
    """Format reranked chunks as structured LLM context string."""
    if not chunks:
        return f"({query_label}: No relevant chunks retrieved)" if query_label else "(No chunks)"

    parts: List[str] = []
    if query_label:
        parts.append(f"### {query_label} ({len(chunks)} chunks after reranking)")

    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        coll = chunk.get("collection", meta.get("collection", "unknown"))
        src = meta.get("source_file", meta.get("source", "unknown"))
        text = chunk["text"].strip().replace("\n", " ")[:450]

        if include_scores and "cross_encoder_score" in chunk:
            score_str = (
                f"| cross-encoder: {chunk['cross_encoder_score']:.3f} "
                f"| bi-encoder-dist: {chunk.get('bi_encoder_distance', chunk.get('distance', 0)):.3f}"
            )
        else:
            dist = chunk.get("distance", 0)
            score_str = f"| distance: {dist:.3f}" if isinstance(dist, (int, float)) else ""

        parts.append(
            f"  [{i}] Collection: {coll} | File: {src} {score_str}\n       {text}"
        )

    return "\n\n".join(parts)


def build_risk_classifier_context(
    disruption_type: str,
    order_region: Optional[str],
    export_control_elevated: bool = False,
) -> str:
    """
    Issue RAG queries for Risk Classifier Signal 3 (two-stage retrieve + rerank).

    Returns formatted context string for the LLM user message.
    """
    sections: List[str] = []

    q1_results = retrieve_and_rerank(
        query=(
            f"supply chain disruption {disruption_type} electronics semiconductor "
            f"historical impact risk score {order_region or ''}"
        ),
        collections=["historical_precedents"],
        bi_encoder_top_n=10,
        rerank_top_k=3,
    )
    sections.append(format_rag_chunks(q1_results, "Historical Precedents"))

    if export_control_elevated:
        q2_results = retrieve_and_rerank(
            query=(
                "export control semiconductor restriction BIS entity list "
                "alternate sourcing compliance trade restriction"
            ),
            collections=["export_control_corpus"],
            bi_encoder_top_n=8,
            rerank_top_k=2,
        )
        sections.append(format_rag_chunks(q2_results, "Export Control Compliance"))
    else:
        sections.append("### Export Control: not queried (ECL below threshold)")

    return "\n\n" + "\n\n".join(sections)


def build_mitigation_context(
    disruption_type: str,
    order_region: Optional[str],
    risk_label: str,
    export_control_elevated: bool = False,
) -> str:
    """
    Issue three RAG queries for Mitigation Agent L7 (two-stage retrieve + rerank).
    """
    sections: List[str] = []

    q1 = retrieve_and_rerank(
        query=(
            f"supply chain {disruption_type} {order_region or ''} "
            f"{risk_label} risk mitigation response historical"
        ),
        collections=["historical_precedents"],
        bi_encoder_top_n=10,
        rerank_top_k=4,
    )
    sections.append(format_rag_chunks(q1, "Historical Precedents & Mitigation"))

    if export_control_elevated:
        q2 = retrieve_and_rerank(
            query="BIS export control semiconductor restriction compliance sourcing",
            collections=["export_control_corpus"],
            bi_encoder_top_n=8,
            rerank_top_k=2,
        )
        sections.append(format_rag_chunks(q2, "Export Control Compliance"))

    q3 = retrieve_and_rerank(
        query=(
            "India semiconductor mission ISM PLI electronics Dixon Tata "
            "Foxconn Wistron Kaynes alternate sourcing domestic procurement"
        ),
        collections=["india_sourcing_corpus"],
        bi_encoder_top_n=8,
        rerank_top_k=3,
    )
    sections.append(format_rag_chunks(q3, "India Sourcing (ISM/PLI)"))

    return "\n\n" + "\n\n".join(sections)
