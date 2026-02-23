"""
nfl/ingest/pull_injuries.py — Pull current NFL injury report from SDIO into nfl.db.

Usage:
    python nfl/ingest/pull_injuries.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.sdio_client import SDIOClient
from shared.db import get_conn, create_schema


def pull_nfl_injuries():
    create_schema("nfl")
    conn = get_conn("nfl")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS injuries (
            player_id     INTEGER PRIMARY KEY,
            player_name   TEXT,
            team          TEXT,
            position      TEXT,
            status        TEXT,
            injury        TEXT,
            fetched_at    TEXT
        )
    """)
    conn.commit()

    client = SDIOClient("nfl")
    print("[nfl/ingest/injuries] Fetching injury report ...")
    injuries = client.nfl_injuries()

    if not injuries:
        print("[nfl/ingest/injuries] No injuries returned.")
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
            fetched_at,
        )
        for p in injuries
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO injuries VALUES (?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    print(f"[nfl/ingest/injuries] {len(rows)} injury records written.")


if __name__ == "__main__":
    pull_nfl_injuries()
