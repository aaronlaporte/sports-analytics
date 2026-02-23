"""
nfl/ingest/pull_players.py — Pull NFL player roster from SDIO.

Saves to nfl.db players table. Useful for building prediction pool.

Usage:
    python nfl/ingest/pull_players.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.sdio_client import SDIOClient
from shared.db import get_conn, create_schema


def pull_nfl_players():
    create_schema("nfl")
    conn = get_conn("nfl")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id     INTEGER PRIMARY KEY,
            player_name   TEXT,
            team          TEXT,
            position      TEXT,
            status        TEXT,
            jersey        INTEGER,
            fetched_at    TEXT
        )
    """)
    conn.commit()

    client = SDIOClient("nfl")
    print("[nfl/ingest/players] Fetching player roster ...")
    players = client.nfl_players()

    if not players:
        print("[nfl/ingest/players] No players returned.")
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
            p.get("Jersey"),
            fetched_at,
        )
        for p in players
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO players VALUES (?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    print(f"[nfl/ingest/players] {len(rows)} players written.")


if __name__ == "__main__":
    pull_nfl_players()
