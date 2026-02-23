"""
nfl/odds/pull_live_odds.py — Pull live NFL player prop odds from The Odds API.

Fetches player_anytime_touchdown and player_reception_yards for this week's games.
Each game = 1 API request. Only run on-demand to conserve API quota (500/month).

Usage:
    python nfl/odds/pull_live_odds.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.odds_client import OddsClient
from shared.db import get_conn, create_schema


def pull_nfl_odds():
    create_schema("nfl")
    client = OddsClient()

    print("[nfl/odds] Fetching this week's NFL games ...")
    games = client.nfl_games()

    if not games:
        print("[nfl/odds] No NFL games found (offseason?).")
        return

    print(f"[nfl/odds] {len(games)} games. Pulling player props ...")
    if client.requests_remaining is not None:
        print(f"[nfl/odds] API requests remaining: {client.requests_remaining}")

    conn = get_conn("nfl")
    fetched_at = datetime.now(timezone.utc).isoformat()
    total = 0

    for game in games:
        event_id  = game["id"]
        home_team = game["home_team"]
        away_team = game["away_team"]

        try:
            props = client.get_nfl_props(event_id)
        except Exception as e:
            print(f"[nfl/odds]   WARN {away_team} @ {home_team}: {e}")
            continue

        if not props:
            print(f"[nfl/odds]   {away_team} @ {home_team}: no props yet")
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
        total += len(rows)
        print(f"[nfl/odds]   {away_team} @ {home_team}: {len(rows)} prop lines")

    conn.close()
    print(f"[nfl/odds] Done. {total} prop lines written to nfl.db")


if __name__ == "__main__":
    pull_nfl_odds()
