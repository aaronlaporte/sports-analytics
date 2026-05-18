"""
Re-exports OddsClient from nomadata_toolkit.
Key is resolved from keys/odds_key.txt or ODDS_API_KEY env var (shared/config.py pattern).
All existing callers continue to work unchanged.
"""
from shared.config import get_odds_key
from nomadata_toolkit.sports.odds import (
    OddsClient as _Base, OddsError,
    ODDS_SPORT_NBA, ODDS_SPORT_MLB, ODDS_SPORT_NFL,
    NBA_PROP_MARKETS, MLB_PROP_MARKETS, NFL_PROP_MARKETS,
)


class OddsClient(_Base):
    def __init__(self):
        super().__init__(api_key=get_odds_key())


__all__ = [
    "OddsClient", "OddsError",
    "ODDS_SPORT_NBA", "ODDS_SPORT_MLB", "ODDS_SPORT_NFL",
    "NBA_PROP_MARKETS", "MLB_PROP_MARKETS", "NFL_PROP_MARKETS",
]
