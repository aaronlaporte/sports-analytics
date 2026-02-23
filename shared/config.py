"""
shared/config.py — Central config for sports-analytics platform.

API keys are loaded from keys/ directory (gitignored).
Override any value with environment variables.
"""

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── API Keys ──────────────────────────────────────────────────────────────────

def _read_key(filename: str, env_var: str) -> str:
    if val := os.environ.get(env_var):
        return val.strip()
    path = ROOT / "keys" / filename
    if path.exists():
        return path.read_text().strip()
    raise FileNotFoundError(
        f"API key not found. Set {env_var} env var or create keys/{filename}"
    )

def get_sdio_key() -> str:
    return _read_key("sdio_key.txt", "SDIO_API_KEY")

def get_odds_key() -> str:
    return _read_key("odds_key.txt", "ODDS_API_KEY")

# ── Data Paths ────────────────────────────────────────────────────────────────

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

MLB_DB = DATA_DIR / "mlb.db"
NBA_DB = DATA_DIR / "nba.db"
NFL_DB = DATA_DIR / "nfl.db"

# ── SportsDataIO ──────────────────────────────────────────────────────────────

SDIO_BASE = "https://api.sportsdata.io/v3"

SDIO_NFL_BASE  = f"{SDIO_BASE}/nfl"
SDIO_MLB_BASE  = f"{SDIO_BASE}/mlb"
SDIO_NBA_BASE  = f"{SDIO_BASE}/nba"

# ── The Odds API ──────────────────────────────────────────────────────────────

ODDS_BASE = "https://api.the-odds-api.com/v4"

ODDS_SPORT_NBA = "basketball_nba"
ODDS_SPORT_MLB = "baseball_mlb"
ODDS_SPORT_NFL = "americanfootball_nfl"

# Player prop markets by sport
NBA_PROP_MARKETS = ["player_points", "player_rebounds", "player_assists", "player_threes"]
MLB_PROP_MARKETS = ["batter_hits", "pitcher_strikeouts"]
NFL_PROP_MARKETS = ["player_anytime_touchdown", "player_reception_yards"]

ODDS_REGIONS = "us"
ODDS_FORMAT  = "american"

# ── Seasons ───────────────────────────────────────────────────────────────────

MLB_CURRENT_SEASON = os.environ.get("MLB_SEASON", "2025")
NBA_CURRENT_SEASON = os.environ.get("NBA_SEASON", "2025")  # 2024-25 season
NFL_CURRENT_SEASON = os.environ.get("NFL_SEASON", "2025")

NBA_SEASON_TYPES = [1, 2]   # 1=Regular, 2=Playoffs
NFL_SEASON_TYPES = ["REG", "POST"]

# ── Modeling ──────────────────────────────────────────────────────────────────

RANDOM_STATE = 42

# Rolling window sizes for feature engineering
ROLLING_WINDOWS = [3, 5, 10, 20]

# Minimum games played to be included in predictions
MIN_GAMES_MLB = 10
MIN_GAMES_NBA = 5
MIN_GAMES_NFL = 3
