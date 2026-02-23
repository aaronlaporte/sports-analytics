"""
nba/odds/pull_live_odds.py — Pull live NBA player prop odds from The Odds API.

Fetches player_points, player_rebounds, player_assists, player_threes
for tonight's games. Each game = 1 API request.

Usage:
    python nba/odds/pull_live_odds.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.odds_client import OddsClient
from shared.db import get_conn, create_schema


def pull_nba_odds():
    create_schema("nba")
    client = OddsClient()

    print("[nba/odds] Fetching tonight's NBA games ...")
    games = client.nba_games()

    if not games:
        print("[nba/odds] No NBA games found.")
        return

    print(f"[nba/odds] {len(games)} games. Pulling player props ...")
    if client.requests_remaining is not None:
        print(f"[nba/odds] API requests remaining: {client.requests_remaining}")

    conn = get_conn("nba")
    fetched_at = datetime.now(timezone.utc).isoformat()
    total = 0

    for game in games:
        event_id  = game["id"]
        home_team = game["home_team"]
        away_team = game["away_team"]

        try:
            props = client.get_nba_props(event_id)
        except Exception as e:
            print(f"[nba/odds]   WARN {away_team} @ {home_team}: {e}")
            continue

        if not props:
            print(f"[nba/odds]   {away_team} @ {home_team}: no props yet")
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
        print(f"[nba/odds]   {away_team} @ {home_team}: {len(rows)} prop lines")

    conn.close()
    print(f"[nba/odds] Done. {total} prop lines written to nba.db")


if __name__ == "__main__":
    pull_nba_odds()
