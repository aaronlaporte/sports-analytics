"""
mlb/ingest/pull_schedules.py — Pull today's MLB schedule and lineup status from SDIO.

Writes to mlb.db → schedules table including LineupConfirmed flag and
confirmed starting pitcher IDs. Run shortly before first pitch (lineup
confirmation typically posts 1-2 hours before game time).

Usage:
    python mlb/ingest/pull_schedules.py
    python mlb/ingest/pull_schedules.py --date 2025-04-15
"""

import argparse
import sys
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.sdio_client import SDIOClient
from shared.db import get_conn, create_schema


def pull_schedules(game_date: str):
    create_schema("mlb")
    client = SDIOClient("mlb")

    print(f"[schedules] Fetching MLB games for {game_date} ...")
    games = client.mlb_games_by_date(game_date)

    if not games:
        print(f"[schedules] No games found for {game_date}")
        return

    conn = get_conn("mlb")
    upserted = 0

    for g in games:
        game_id = g.get("GameID")
        if not game_id:
            continue

        lineup_confirmed = 1 if g.get("LineupConfirmed") else 0

        # Attempt to get confirmed pitcher IDs from lineups endpoint
        home_pitcher_id = None
        away_pitcher_id = None
        if lineup_confirmed:
            try:
                lineup = client.mlb_lineups(game_id)
                if isinstance(lineup, dict):
                    home_pitcher_id = lineup.get("HomeStartingPitcherID")
                    away_pitcher_id = lineup.get("AwayStartingPitcherID")
            except Exception:
                pass

        conn.execute("""
            INSERT OR REPLACE INTO schedules
                (game_id, game_date, home_team, away_team, status,
                 lineup_confirmed, home_pitcher_id, away_pitcher_id)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            game_id,
            g.get("Day", game_date)[:10],
            g.get("HomeTeam"),
            g.get("AwayTeam"),
            g.get("Status"),
            lineup_confirmed,
            home_pitcher_id,
            away_pitcher_id,
        ))
        upserted += 1

    conn.commit()
    conn.close()

    confirmed = sum(1 for g in games if g.get("LineupConfirmed"))
    print(f"[schedules] {upserted} games upserted, {confirmed} with confirmed lineups")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().strftime("%Y-%m-%d"))
    args = parser.parse_args()
    pull_schedules(args.date)
