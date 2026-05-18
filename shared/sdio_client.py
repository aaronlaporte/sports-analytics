"""
Re-exports SDIOClient from nomadata_toolkit.
Key is resolved from keys/sdio_key.txt or SDIO_API_KEY env var (shared/config.py pattern).
All existing callers (SDIOClient("mlb"), etc.) continue to work unchanged.
"""
from shared.config import get_sdio_key
from nomadata_toolkit.sports.sdio import SDIOClient as _Base, SDIOError


class SDIOClient(_Base):
    def __init__(self, sport: str):
        super().__init__(sport=sport, api_key=get_sdio_key())


__all__ = ["SDIOClient", "SDIOError"]
