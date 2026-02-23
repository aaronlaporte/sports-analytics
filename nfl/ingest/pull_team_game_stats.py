"""
nfl/ingest/pull_team_game_stats.py — Pull NFL team season stats from SDIO into nfl.db.

Used to compute opponent scoring context for feature engineering.

Usage:
    python nfl/ingest/pull_team_game_stats.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.sdio_client import SDIOClient
from shared.db import get_conn, create_schema
from shared.config import NFL_CURRENT_SEASON


def pull_nfl_team_stats():
    create_schema("nfl")
    conn = get_conn("nfl")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_season_stats (
            team          TEXT,
            season        TEXT,
            touchdowns    REAL,
            rush_yards    REAL,
            pass_yards    REAL,
            points_per_game REAL,
            td_per_game   REAL,
            PRIMARY KEY (team, season)
        )
    """)
    conn.commit()

    client = SDIOClient("nfl")
    seasons = [str(int(NFL_CURRENT_SEASON) - 2),
               str(int(NFL_CURRENT_SEASON) - 1),
               NFL_CURRENT_SEASON]
    total = 0

    for season in seasons:
        print(f"[nfl/ingest/team] Pulling team stats for {season} ...")
        try:
            stats = client.nfl_team_season_stats(season)
        except Exception as e:
            print(f"[nfl/ingest/team]   WARN {season}: {e}")
            continue

        if not stats:
            continue

        rows = [
            (
                t.get("Team"),
                season,
                t.get("Touchdowns"),
                t.get("RushingYards"),
                t.get("PassingYards"),
                t.get("PointsPerGame"),
                (t.get("Touchdowns") or 0) / max(t.get("Games") or 1, 1),
            )
            for t in stats
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO team_season_stats VALUES (?,?,?,?,?,?,?)", rows
        )
        conn.commit()
        total += len(rows)
        print(f"[nfl/ingest/team]   {season}: {len(rows)} teams")

    conn.close()
    print(f"[nfl/ingest/team] Done. {total} team-season rows written.")


if __name__ == "__main__":
    pull_nfl_team_stats()
