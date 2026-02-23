"""
shared/odds_client.py — The Odds API client.

IMPORTANT: Player props are ONLY available via the event-level endpoint:
  /v4/sports/{sport}/events/{event_id}/odds
The top-level /odds/ endpoint does NOT support player prop markets.

Free tier: 500 requests/month. Pull on-demand only, not on cron.
Each game = 1 request for player props.
"""

import requests
from typing import Optional
from shared.config import (
    get_odds_key, ODDS_BASE, ODDS_REGIONS, ODDS_FORMAT,
    ODDS_SPORT_NBA, ODDS_SPORT_MLB, ODDS_SPORT_NFL,
    NBA_PROP_MARKETS, MLB_PROP_MARKETS, NFL_PROP_MARKETS,
)


class OddsError(Exception):
    pass


class OddsClient:
    def __init__(self):
        self.key = get_odds_key()
        self._remaining = None  # tracked from response headers

    def _get(self, path: str, **params) -> dict | list:
        url = f"{ODDS_BASE}/{path}"
        params["apiKey"] = self.key

        resp = requests.get(url, params=params, timeout=15)

        # Track quota from headers
        if "x-requests-remaining" in resp.headers:
            self._remaining = int(resp.headers["x-requests-remaining"])

        if not resp.ok:
            data = resp.json() if resp.content else {}
            raise OddsError(
                f"Odds API {resp.status_code}: {data.get('message', resp.text[:200])}"
            )
        return resp.json()

    @property
    def requests_remaining(self) -> Optional[int]:
        """Requests left this month. None until first call is made."""
        return self._remaining

    # ── Game List (h2h only — 1 request, gets all games + event IDs) ──────────

    def get_games(self, sport: str) -> list[dict]:
        """
        Get tonight's/upcoming games with event IDs.
        sport: ODDS_SPORT_NBA, ODDS_SPORT_MLB, ODDS_SPORT_NFL
        Returns list of: {id, home_team, away_team, commence_time}
        Cost: 1 request.
        """
        data = self._get(
            f"sports/{sport}/odds/",
            regions=ODDS_REGIONS,
            markets="h2h",
            oddsFormat=ODDS_FORMAT,
        )
        return [
            {
                "id": g["id"],
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "commence_time": g["commence_time"],
            }
            for g in data
        ]

    # ── Player Props (1 request per game) ─────────────────────────────────────

    def get_player_props(self, sport: str, event_id: str, markets: list[str]) -> list[dict]:
        """
        Fetch player prop lines for a single game.
        Cost: 1 request per call.

        Returns flat list of:
          {player, market, name (Over/Under), line, price, bookmaker}
        """
        data = self._get(
            f"sports/{sport}/events/{event_id}/odds",
            regions=ODDS_REGIONS,
            markets=",".join(markets),
            oddsFormat=ODDS_FORMAT,
        )

        rows = []
        for bookmaker in data.get("bookmakers", []):
            book_key = bookmaker["key"]
            for market in bookmaker.get("markets", []):
                market_key = market["key"]
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "event_id":  event_id,
                        "bookmaker": book_key,
                        "market":    market_key,
                        "player":    outcome.get("description", ""),
                        "name":      outcome.get("name", ""),    # Over / Under
                        "line":      outcome.get("point"),        # e.g. 25.5
                        "price":     outcome.get("price"),        # American odds
                    })
        return rows

    # ── Convenience wrappers by sport ─────────────────────────────────────────

    def get_nba_props(self, event_id: str) -> list[dict]:
        """Player props for one NBA game. Cost: 1 request."""
        return self.get_player_props(ODDS_SPORT_NBA, event_id, NBA_PROP_MARKETS)

    def get_mlb_props(self, event_id: str) -> list[dict]:
        """Player props for one MLB game. Cost: 1 request."""
        return self.get_player_props(ODDS_SPORT_MLB, event_id, MLB_PROP_MARKETS)

    def get_nfl_props(self, event_id: str) -> list[dict]:
        """Player props for one NFL game. Cost: 1 request."""
        return self.get_player_props(ODDS_SPORT_NFL, event_id, NFL_PROP_MARKETS)

    def nba_games(self) -> list[dict]:
        return self.get_games(ODDS_SPORT_NBA)

    def mlb_games(self) -> list[dict]:
        return self.get_games(ODDS_SPORT_MLB)

    def nfl_games(self) -> list[dict]:
        return self.get_games(ODDS_SPORT_NFL)
