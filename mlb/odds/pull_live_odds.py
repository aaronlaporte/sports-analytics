"""
mlb/odds/pull_live_odds.py — Pull live MLB player prop odds from The Odds API.

Fetches batter_hits and pitcher_strikeouts lines for today's games.
Each game costs 1 API request. Pull on-demand only to conserve 500/month quota.

Usage:
    python mlb/odds/pull_live_odds.py
    python mlb/odds/pull_live_odds.py --date 2025-04-15
"""

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.odds_client import OddsClient
from shared.db import get_conn, create_schema


def pull_mlb_odds(game_date: str | None = None):
    create_schema("mlb")
    client = OddsClient()

    print("[odds] Fetching today's MLB games ...")
    games = client.mlb_games()

    if not games:
        print("[odds] No MLB games found (offseason or no games today).")
        return

    print(f"[odds] {len(games)} games found. Pulling player props ...")
    if client.requests_remaining is not None:
        print(f"[odds] Requests remaining this month: {client.requests_remaining}")

    conn = get_conn("mlb")
    fetched_at = datetime.now(timezone.utc).isoformat()
    total_rows = 0

    for game in games:
        event_id  = game["id"]
        home_team = game["home_team"]
        away_team = game["away_team"]

        try:
            props = client.get_mlb_props(event_id)
        except Exception as e:
            print(f"[odds]   WARN: {away_team} @ {home_team} — {e}")
            continue

        if not props:
            print(f"[odds]   {away_team} @ {home_team}: no props available yet")
            continue

        rows = [
            (fetched_at, event_id, home_team, away_team,
             p["bookmaker"], p["market"], p["player"],
             p["name"], p["line"], p["price"])
            for p in props
        ]
        conn.executemany("""
            INSERT OR REPLACE INTO live_odds
                (fetched_at, event_id, home_team, away_team,
                 bookmaker, market, player, name, line, price)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, rows)
        conn.commit()
        total_rows += len(rows)
        print(f"[odds]   {away_team} @ {home_team}: {len(rows)} prop lines stored")

    conn.close()
    print(f"[odds] Done. {total_rows} total prop lines written to mlb.db")
    if client.requests_remaining is not None:
        print(f"[odds] Requests remaining: {client.requests_remaining}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Game date (informational only)")
    args = parser.parse_args()
    pull_mlb_odds(args.date)
