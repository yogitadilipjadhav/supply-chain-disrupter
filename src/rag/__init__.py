"""
src/rag — Retrieval-augmented generation for the supply-chain pipeline.

Modules:
  utils        — ChromaDB client, embedding model, monolithic corpus build/query
  collections  — Named collection ingest (historical / export / India sourcing)
  retriever    — Two-stage retrieve (bi-encoder) + rerank (cross-encoder)
  agent        — News-signal fallback via RAG query
"""

from src.rag.agent import build_news_signals, query_news_signals
from src.rag.collections import (
    COLLECTION_NAMES,
    build_collection,
    query_collection,
    query_multi_collection,
)
from src.rag.retriever import (
    build_mitigation_context,
    build_risk_classifier_context,
    retrieve_and_rerank,
    rerank_results,
)
from src.rag.utils import (
    DEFAULT_COLLECTION_NAME,
    build_chroma_complete,
    build_rag_corpus_complete,
    get_chroma_client,
    get_embedding_model,
    query_chroma_rag,
    reset_chroma_client,
    resolve_embedding_model_name,
)

__all__ = [
    "COLLECTION_NAMES",
    "DEFAULT_COLLECTION_NAME",
    "build_chroma_complete",
    "build_collection",
    "build_mitigation_context",
    "build_news_signals",
    "build_rag_corpus_complete",
    "build_risk_classifier_context",
    "get_chroma_client",
    "get_embedding_model",
    "query_chroma_rag",
    "reset_chroma_client",
    "resolve_embedding_model_name",
    "query_collection",
    "query_multi_collection",
    "query_news_signals",
    "retrieve_and_rerank",
    "rerank_results",
]
