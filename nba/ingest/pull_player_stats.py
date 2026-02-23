"""
nba/ingest/pull_player_stats.py — Pull NBA player game stats from SDIO into nba.db.

Pulls daily box scores for a date range and upserts into player_game_stats.
Run nightly after games complete (~midnight ET).

Usage:
    python nba/ingest/pull_player_stats.py
    python nba/ingest/pull_player_stats.py --start 2024-10-22 --end 2025-04-13
"""

import argparse
import sys
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.sdio_client import SDIOClient
from shared.db import get_conn, create_schema

# 2024-25 NBA regular season start
NBA_SEASON_START = "2024-10-22"


def sdio_date(d: date) -> str:
    """Convert date → SDIO format: 'YYYY-MON-DD' (e.g. '2025-FEB-20')"""
    return d.strftime("%Y-%b-%d").upper()


def load_existing_dates(conn) -> set:
    rows = conn.execute("SELECT DISTINCT game_date FROM player_game_stats").fetchall()
    return {r["game_date"] for r in rows}


def pull_player_stats(start: str, end: str):
    create_schema("nba")
    conn    = get_conn("nba")
    client  = SDIOClient("nba")
    existing = load_existing_dates(conn)

    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").date()

    current = start_dt
    total_rows = 0

    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")

        if date_str in existing:
            current += timedelta(days=1)
            continue

        sdio_str = sdio_date(current)
        print(f"[nba/ingest] Pulling {date_str} ...")

        try:
            players = client.nba_player_game_stats_by_date(sdio_str)
        except Exception as e:
            print(f"[nba/ingest]   ERROR: {e}")
            current += timedelta(days=1)
            continue

        if not players:
            current += timedelta(days=1)
            continue

        rows = []
        for p in players:
            rows.append((
                p.get("StatID"),
                p.get("PlayerID"),
                p.get("Name"),
                p.get("Team"),
                p.get("Opponent"),
                p.get("GameID"),
                date_str,
                p.get("Season"),
                p.get("SeasonType"),
                p.get("HomeOrAway"),
                1 if p.get("Started") else 0,
                p.get("Minutes") or 0,
                p.get("Points"),
                p.get("Rebounds"),
                p.get("Assists"),
                p.get("Steals"),
                p.get("BlockedShots"),
                p.get("Turnovers"),
                p.get("PersonalFouls"),
                p.get("ThreePointersMade"),
                p.get("ThreePointersAttempted"),
                p.get("FieldGoalsMade"),
                p.get("FieldGoalsAttempted"),
                p.get("FieldGoalsPercentage"),
                p.get("FreeThrowsMade"),
                p.get("FreeThrowsAttempted"),
                p.get("OffensiveRebounds"),
                p.get("DefensiveRebounds"),
                p.get("UsageRatePercentage"),
                p.get("TrueShootingPercentage"),
                p.get("PlayerEfficiencyRating"),
                p.get("PlusMinus"),
                p.get("FantasyPointsDraftKings"),
                p.get("FantasyPointsFanDuel"),
                p.get("DraftKingsSalary"),
                p.get("FanDuelSalary"),
                p.get("YahooSalary"),
                p.get("OpponentRank"),
                p.get("OpponentPositionRank"),
                p.get("InjuryStatus"),
                1 if p.get("LineupConfirmed") else 0,
                1 if p.get("IsGameOver") else 0,
            ))

        conn.executemany("""
            INSERT OR REPLACE INTO player_game_stats VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
        conn.commit()
        total_rows += len(rows)
        print(f"[nba/ingest]   {len(rows)} players stored for {date_str}")

        current += timedelta(days=1)

    conn.close()
    print(f"[nba/ingest] Done. Total rows upserted: {total_rows}")


if __name__ == "__main__":
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=NBA_SEASON_START)
    parser.add_argument("--end",   default=yesterday)
    args = parser.parse_args()
    pull_player_stats(args.start, args.end)
