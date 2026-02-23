"""
nfl/ingest/pull_player_game_stats.py — Pull NFL player game stats from SDIO into nfl.db.

Pulls week-by-week player box scores for all configured seasons.
Skips already-ingested season+week combinations.

Usage:
    python nfl/ingest/pull_player_game_stats.py
    python nfl/ingest/pull_player_game_stats.py --seasons 2024REG 2024POST
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.sdio_client import SDIOClient
from shared.db import get_conn, create_schema
from shared.config import NFL_CURRENT_SEASON

# Historical seasons to ingest for model training
DEFAULT_SEASONS = [
    "2022REG", "2022POST",
    "2023REG", "2023POST",
    "2024REG", "2024POST",
    f"{NFL_CURRENT_SEASON}REG",
]


def load_existing(conn) -> set:
    rows = conn.execute(
        "SELECT DISTINCT season || '_' || week AS key FROM player_game_stats"
    ).fetchall()
    return {r["key"] for r in rows}


def pull_nfl_player_game_stats(seasons: list = None):
    create_schema("nfl")
    conn = get_conn("nfl")
    client = SDIOClient("nfl")

    if not seasons:
        seasons = DEFAULT_SEASONS

    existing = load_existing(conn)
    total = 0

    for season in seasons:
        max_weeks = 18 if "REG" in season else 5
        print(f"[nfl/ingest/pgs] {season} ...")

        for week in range(1, max_weeks + 1):
            key = f"{season}_{week}"
            if key in existing:
                continue

            try:
                players = client.nfl_player_game_stats_by_week(season, week)
            except Exception as e:
                print(f"[nfl/ingest/pgs]   WARN {season} W{week}: {e}")
                continue

            if not players:
                continue

            rows = []
            for p in players:
                rush_td = p.get("RushingTouchdowns") or 0
                rec_td  = p.get("ReceivingTouchdowns") or 0
                rows.append((
                    p.get("StatID"),
                    p.get("PlayerID"),
                    p.get("Name"),
                    p.get("Team"),
                    p.get("Opponent"),
                    p.get("GameKey"),
                    season,
                    1 if "REG" in season else 2,
                    week,
                    (p.get("GameDate") or p.get("Date") or "")[:10],
                    p.get("HomeOrAway"),
                    p.get("Position"),
                    p.get("RushingAttempts"),
                    p.get("RushingYards"),
                    rush_td,
                    p.get("Targets"),
                    p.get("Receptions"),
                    p.get("ReceivingYards"),
                    rec_td,
                    rush_td + rec_td,
                    p.get("FantasyPointsDraftKings"),
                    p.get("FantasyPointsFanDuel"),
                    p.get("InjuryStatus"),
                ))

            conn.executemany("""
                INSERT OR REPLACE INTO player_game_stats
                    (stat_id, player_id, player_name, team, opponent, game_key,
                     season, season_type, week, game_date, home_or_away, position,
                     rush_attempts, rush_yards, rush_tds, receiving_targets, receptions,
                     receiving_yards, receiving_tds, total_tds,
                     fantasy_pts_dk, fantasy_pts_fd, injury_status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows)
            conn.commit()
            total += len(rows)
            print(f"[nfl/ingest/pgs]   {season} W{week}: {len(rows)} rows")

    conn.close()
    print(f"[nfl/ingest/pgs] Done. {total} rows written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=None)
    args = parser.parse_args()
    pull_nfl_player_game_stats(args.seasons)
