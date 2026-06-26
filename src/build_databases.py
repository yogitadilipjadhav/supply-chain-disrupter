"""Backward-compatible entry point for ``python -m src.build_databases``."""

from scripts.build_databases import main

if __name__ == "__main__":
    main()
