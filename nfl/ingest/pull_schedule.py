"""
nfl/ingest/pull_schedule.py — Pull NFL schedule from SDIO into nfl.db.

Pulls game scores/schedule for all weeks in the current season.

Usage:
    python nfl/ingest/pull_schedule.py
    python nfl/ingest/pull_schedule.py --seasons 2024REG 2024POST
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.sdio_client import SDIOClient
from shared.db import get_conn, create_schema
from shared.config import NFL_CURRENT_SEASON


def pull_nfl_schedule(seasons: list = None):
    create_schema("nfl")
    conn = get_conn("nfl")
    client = SDIOClient("nfl")

    if not seasons:
        seasons = [f"{NFL_CURRENT_SEASON}REG", f"{NFL_CURRENT_SEASON}POST"]

    total = 0
    for season in seasons:
        # REG = 18 weeks; POST = 4 rounds
        max_weeks = 18 if "REG" in season else 5
        print(f"[nfl/ingest/sched] Pulling {season} ({max_weeks} weeks) ...")

        for week in range(1, max_weeks + 1):
            try:
                games = client.nfl_scores_by_week(season, week)
            except Exception as e:
                print(f"[nfl/ingest/sched]   WARN week {week}: {e}")
                continue

            if not games:
                continue

            rows = [
                (
                    g.get("GameKey"),
                    season,
                    week,
                    (g.get("Date") or "")[:10],
                    g.get("HomeTeam"),
                    g.get("AwayTeam"),
                    g.get("HomeScore"),
                    g.get("AwayScore"),
                    g.get("PointSpread"),
                    g.get("OverUnder"),
                )
                for g in games
            ]
            conn.executemany("""
                INSERT OR REPLACE INTO schedules
                    (game_key, season, week, game_date, home_team, away_team,
                     home_score, away_score, point_spread, over_under)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, rows)
            conn.commit()
            total += len(rows)
            print(f"[nfl/ingest/sched]   Week {week}: {len(rows)} games")

    conn.close()
    print(f"[nfl/ingest/sched] Done. {total} schedule rows written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=None)
    args = parser.parse_args()
    pull_nfl_schedule(args.seasons)
