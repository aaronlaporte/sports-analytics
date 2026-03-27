"""
mlb_insights/utils/db.py -- Database connection and schema helpers.

Provides a single get_connection() that all modules use.  Also contains
the player_pages table DDL and any migration logic.
"""

import logging
import sqlite3
from contextlib import contextmanager

from mlb_insights.config import DB_PATH

logger = logging.getLogger(__name__)

# ── Player Pages schema (new for Phase 3) ────────────────────────────────────

PLAYER_PAGES_DDL = """
CREATE TABLE IF NOT EXISTS player_pages (
    page_date TEXT,
    player_id INTEGER,
    player_name TEXT,
    team TEXT,
    -- Current Form
    season_avg REAL,
    season_hit_pct REAL,
    current_streak INTEGER,
    hits_last_5 REAL,
    hits_last_10 REAL,
    -- Signals
    active_signals_json TEXT,
    -- Prediction History
    last_10_predictions_json TEXT,
    prediction_accuracy_30d REAL,
    -- Composite
    daily_rank INTEGER,
    daily_score REAL,
    PRIMARY KEY (page_date, player_id)
);
"""

CALIBRATION_SUMMARY_DDL = """
CREATE TABLE IF NOT EXISTS calibration_summary (
    summary_date    TEXT,
    metric_type     TEXT,
    window_days     INTEGER,
    value           REAL,
    sample_size     INTEGER,
    PRIMARY KEY (summary_date, metric_type, window_days)
);
"""


def get_connection(readonly: bool = False) -> sqlite3.Connection:
    """Return a sqlite3 connection to the MLB database.

    Args:
        readonly: If True, open in read-only mode (uri connection).

    Returns:
        sqlite3.Connection with Row factory and WAL mode enabled.
    """
    if readonly:
        uri = f"file:{DB_PATH}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def managed_connection(readonly: bool = False):
    """Context manager that yields a connection and commits/closes cleanly.

    Usage::

        with managed_connection() as conn:
            conn.execute("INSERT ...")
    """
    conn = get_connection(readonly=readonly)
    try:
        yield conn
        if not readonly:
            conn.commit()
    except Exception:
        if not readonly:
            conn.rollback()
        raise
    finally:
        conn.close()


def ensure_tables():
    """Create any Phase 3 tables that don't already exist.

    Existing tables (batter_stats, pitcher_stats, daily_leaderboard, etc.)
    are managed by shared/db.py.  This function only creates new tables
    introduced in Phase 3.
    """
    with managed_connection() as conn:
        conn.executescript(PLAYER_PAGES_DDL)
        conn.executescript(CALIBRATION_SUMMARY_DDL)
    logger.info("Phase 3 tables verified.")
