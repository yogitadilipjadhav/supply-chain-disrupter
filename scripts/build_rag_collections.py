"""CLI entry point — delegates to src.rag.collections.main()."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.collections import main

if __name__ == "__main__":
    main()
