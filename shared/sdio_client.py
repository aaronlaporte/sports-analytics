"""
shared/sdio_client.py — SportsDataIO API client.

Wraps all three sports (MLB, NBA, NFL) with rate limiting,
error handling, and a consistent interface.
"""

import time
import requests
from typing import Any
from shared.config import get_sdio_key, SDIO_MLB_BASE, SDIO_NBA_BASE, SDIO_NFL_BASE


class SDIOError(Exception):
    pass


class SDIOClient:
    REQUEST_DELAY = 0.2  # seconds between requests (polite rate limiting)

    def __init__(self, sport: str):
        """
        sport: 'mlb', 'nba', or 'nfl'
        """
        self.key = get_sdio_key()
        self.base = {
            "mlb": SDIO_MLB_BASE,
            "nba": SDIO_NBA_BASE,
            "nfl": SDIO_NFL_BASE,
        }[sport.lower()]
        self._last_request = 0.0

    def _get(self, path: str, **params) -> Any:
        """Make a GET request with rate limiting and error handling."""
        elapsed = time.time() - self._last_request
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)

        url = f"{self.base}/{path}"
        params["key"] = self.key

        resp = requests.get(url, params=params, timeout=15)
        self._last_request = time.time()

        if resp.status_code == 401:
            raise SDIOError(f"Unauthorized: {path} — endpoint may require a paid tier")
        if resp.status_code == 404:
            return []  # No data for this date/week — normal for offseason
        if not resp.ok:
            raise SDIOError(f"SDIO {resp.status_code} on {path}: {resp.text[:200]}")

        return resp.json()

    # ── MLB Endpoints ─────────────────────────────────────────────────────────

    def mlb_games_by_date(self, date: str) -> list:
        """date: 'YYYY-MM-DD' → list of games with lineup confirmation status."""
        return self._get(f"scores/json/GamesByDate/{date}")

    def mlb_lineups(self, game_id: int) -> dict:
        """Confirmed batting order and starting pitchers for a game."""
        return self._get(f"scores/json/Lineups/{game_id}")

    def mlb_player_game_stats_by_date(self, date: str) -> list:
        """All player box scores for games on a given date."""
        return self._get(f"stats/json/PlayerGameStatsByDate/{date}")

    def mlb_players(self) -> list:
        """Full player roster with handedness, position, team."""
        return self._get("scores/json/Players")

    def mlb_injuries(self) -> list:
        """Current injury report."""
        return self._get("scores/json/InjuredPlayers")

    # ── NBA Endpoints ─────────────────────────────────────────────────────────

    def nba_games_by_date(self, date: str) -> list:
        """date: 'YYYY-MMM-DD' (e.g. '2025-FEB-20') → list of games."""
        return self._get(f"scores/json/GamesByDate/{date}")

    def nba_player_game_stats_by_date(self, date: str) -> list:
        """
        All player box scores for games on a given date.
        date: 'YYYY-MMM-DD' (e.g. '2025-FEB-20')
        Returns ~317 rows per active game night with 80+ fields.
        """
        return self._get(f"stats/json/PlayerGameStatsByDate/{date}")

    def nba_players(self) -> list:
        """Full player roster."""
        return self._get("scores/json/Players")

    def nba_injuries(self) -> list:
        """Current injury report."""
        return self._get("scores/json/InjuredPlayers")

    def nba_teams(self) -> list:
        """All 30 NBA teams."""
        return self._get("scores/json/Teams")

    def nba_standings(self, season: str) -> list:
        """Season standings."""
        return self._get(f"scores/json/Standings/{season}")

    # ── NFL Endpoints ─────────────────────────────────────────────────────────

    def nfl_scores_by_week(self, season: str, week: int) -> list:
        """Game scores and schedule for a given season/week."""
        return self._get(f"scores/json/ScoresByWeek/{season}/{week}")

    def nfl_players(self) -> list:
        """Full player roster."""
        return self._get("scores/json/Players")

    def nfl_player_game_stats_by_week(self, season: str, week: int) -> list:
        """Player game stats for a given season/week."""
        return self._get(f"stats/json/PlayerGameStatsByWeek/{season}/{week}")

    def nfl_team_season_stats(self, season: str) -> list:
        """Team season-level stats."""
        return self._get(f"stats/json/TeamSeasonStats/{season}")

    def nfl_injuries(self) -> list:
        """Current injury report."""
        return self._get("scores/json/InjuredPlayers")
