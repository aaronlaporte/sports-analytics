"""
mlb/ingest/pull_injuries.py — Pull current MLB injury report from SDIO into mlb.db.

Stores latest injury report in the injuries table (replace on refresh).

Usage:
    python mlb/ingest/pull_injuries.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.sdio_client import SDIOClient
from shared.db import get_conn, create_schema


def pull_injuries():
    create_schema("mlb")
    conn = get_conn("mlb")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS injuries (
            player_id     INTEGER PRIMARY KEY,
            player_name   TEXT,
            team          TEXT,
            position      TEXT,
            status        TEXT,
            injury        TEXT,
            start_date    TEXT,
            fetched_at    TEXT
        )
    """)
    conn.commit()

    client = SDIOClient("mlb")
    print("[mlb/ingest] Fetching injury report ...")
    injuries = client.mlb_injuries()

    if not injuries:
        print("[mlb/ingest] No injuries returned.")
        conn.close()
        return

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            p.get("PlayerID"),
            p.get("Name"),
            p.get("Team"),
            p.get("Position"),
            p.get("Status"),
            p.get("Injury"),
            p.get("StartDate"),
            fetched_at,
        )
        for p in injuries
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO injuries VALUES (?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    print(f"[mlb/ingest] {len(rows)} injury records written.")


if __name__ == "__main__":
    pull_injuries()
