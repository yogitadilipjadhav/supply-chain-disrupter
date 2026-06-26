"""
util_inspect_db_schema.py
==========================
Purpose  : Quick diagnostic utility — prints the schema, row counts, and
           column names for every table and view in outputs/supply_chain.db.

When to use
-----------
Run this script to confirm the DB structure after any of:
  • Running scripts/build_databases.py  (builds the real 5,459-row DB)
  • Running fixture_spec_conformant_db.py  (builds the synthetic fixture)
  • Running any QA test that calls ensure_risk_classification_table()

This script is read-only — it makes no changes to the database.

Usage
-----
  python evaluation/util_inspect_db_schema.py
"""

import sqlite3
import sys
import os

sys.path.insert(0, ".")  # allow imports from project root

DB_PATH = "outputs/supply_chain.db"


def inspect_db(db_path: str) -> None:
    """
    Connect to the SQLite database and print a summary of every object
    (table or view) including its row count and first 10 column names.
    """
    if not os.path.exists(db_path):
        print(f"ERROR | Database not found: {db_path}")
        print("       Run build_databases.py or fixture_spec_conformant_db.py first.")
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    # Fetch all tables and views sorted by type then name
    objects = conn.execute(
        """
        SELECT type, name
        FROM sqlite_master
        WHERE type IN ('table', 'view')
        ORDER BY type, name
        """
    ).fetchall()

    print(f"=== DB Schema Inspector: {db_path} ===")
    print(f"Objects found: {len(objects)}")
    print()

    for obj_type, obj_name in objects:
        # Row count — VIEWs support COUNT(*) just like tables
        count = conn.execute(f"SELECT COUNT(*) FROM {obj_name}").fetchone()[0]

        # Column names via PRAGMA (returns nothing for VIEWs in some SQLite versions,
        # so fall back to a SELECT with LIMIT 0 if needed)
        col_rows = conn.execute(f"PRAGMA table_info({obj_name})").fetchall()
        col_names = [row[1] for row in col_rows]

        print(f"  [{obj_type.upper()}] {obj_name}")
        print(f"    rows    : {count:,}")
        if col_names:
            preview = col_names[:10]
            suffix  = f" ... (+{len(col_names) - 10} more)" if len(col_names) > 10 else ""
            print(f"    columns : {preview}{suffix}")
        print()

    conn.close()


if __name__ == "__main__":
    inspect_db(DB_PATH)
