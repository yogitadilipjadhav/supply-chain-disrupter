from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/build_databases.py` from project root (Colab, CLI)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.etl_loader import get_sqlite_stats, load_excel_into_sqlite
from src.rag.utils import build_rag_corpus_complete, query_chroma_rag


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build electronics SQLite database and ChromaDB stores."
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Upsert Chroma content instead of clearing the existing collection.",
    )
    args = parser.parse_args()

    inserted = load_excel_into_sqlite(flush_existing=True)
    print(f"SQLite: loaded {inserted:,} Lite Master orders")
    print(json.dumps(get_sqlite_stats(), indent=2))

    chroma_summary = build_rag_corpus_complete(
        flush_existing=not args.no_rebuild
    )
    print("ChromaDB:")
    print(json.dumps(chroma_summary, indent=2))

    smoke_queries = [
        "semiconductor factory shutdown and chip shortage risk",
        "critical electronics supply disruption mitigation",
        "what field contains safety stock and lead time",
        "Red Sea route disruption shipping risk",
        "semiconductor production vulnerabilities and geographic concentration",
    ]
    for query in smoke_queries:
        hits = query_chroma_rag(query, n_results=1)
        title = hits[0]["metadata"].get("type") if hits else "NO HIT"
        print(f"RAG smoke test: {query!r} -> {title}")


if __name__ == "__main__":
    main()
