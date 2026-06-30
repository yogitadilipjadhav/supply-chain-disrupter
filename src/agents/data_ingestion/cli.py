"""CLI for the Data Ingestion agent's live feeds.

Examples (run from project root):

    python -m src.agents.data_ingestion.cli            # weather + news
    python -m src.agents.data_ingestion.cli --weather  # weather only
    python -m src.agents.data_ingestion.cli --news     # news only (GDELT)
    python -m src.agents.data_ingestion.cli --news --rss   # news from GDELT + RSS feeds
"""

from __future__ import annotations

import argparse
import json

from src.agents.data_ingestion.live_ingest import ingest_news, ingest_weather


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest live weather (Open-Meteo) and news (GDELT/RSS) into SQLite."
    )
    parser.add_argument("--weather", action="store_true", help="Ingest weather only.")
    parser.add_argument("--news", action="store_true", help="Ingest news only.")
    parser.add_argument(
        "--rss", action="store_true", help="Include RSS feeds in news ingestion."
    )
    parser.add_argument(
        "--timespan", default="3d", help="GDELT lookback window, e.g. 1d, 3d, 1w."
    )
    parser.add_argument(
        "--max-records", type=int, default=50, help="Max GDELT articles per query."
    )
    args = parser.parse_args()

    run_all = not (args.weather or args.news)

    if args.weather or run_all:
        print("Weather ingestion:")
        print(json.dumps(ingest_weather(), indent=2))

    if args.news or run_all:
        print("News ingestion:")
        print(
            json.dumps(
                ingest_news(
                    max_records=args.max_records,
                    timespan=args.timespan,
                    use_rss=args.rss,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
