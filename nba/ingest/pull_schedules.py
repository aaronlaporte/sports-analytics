"""
nba/ingest/pull_schedules.py — Pull NBA schedules from SDIO into nba.db.

Pulls all games for the current season and upserts into the schedules table.

Usage:
    python nba/ingest/pull_schedules.py
"""

import sys
from datetime import datetime, timedelta, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.sdio_client import SDIOClient
from shared.db import get_conn, create_schema


def sdio_date(d: date) -> str:
    return d.strftime("%Y-%b-%d").upper()


def pull_nba_schedules():
    create_schema("nba")
    conn = get_conn("nba")
    client = SDIOClient("nba")

    # Pull ~90 days of schedule: 60 past + 30 future
    today = datetime.today().date()
    start = today - timedelta(days=60)
    end   = today + timedelta(days=30)

    current = start
    total = 0
    seen_dates = set()

    while current <= end:
        sdio_str = sdio_date(current)
        if sdio_str in seen_dates:
            current += timedelta(days=1)
            continue
        seen_dates.add(sdio_str)

        try:
            games = client.nba_games_by_date(sdio_str)
        except Exception as e:
            print(f"[nba/ingest/sched] WARN {sdio_str}: {e}")
            current += timedelta(days=1)
            continue

        if games:
            rows = [
                (
                    g.get("GameID"),
                    g.get("Day", "")[:10] if g.get("Day") else current.isoformat(),
                    g.get("Season"),
                    g.get("SeasonType"),
                    g.get("HomeTeam"),
                    g.get("AwayTeam"),
                    g.get("HomeTeamScore"),
                    g.get("AwayTeamScore"),
                    g.get("Status"),
                )
                for g in games
            ]
            conn.executemany("""
                INSERT OR REPLACE INTO schedules
                    (game_id, game_date, season, season_type, home_team, away_team,
                     home_score, away_score, status)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, rows)
            conn.commit()
            total += len(rows)

        current += timedelta(days=1)

    conn.close()
    print(f"[nba/ingest/sched] Done. {total} schedule rows written.")


if __name__ == "__main__":
    pull_nba_schedules()
