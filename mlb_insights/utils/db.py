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

HR_FEATURES_DDL = """
CREATE TABLE IF NOT EXISTS hr_features (
    feature_date TEXT,
    batter_id INTEGER,
    barrel_rate REAL,
    avg_exit_velo REAL,
    hr_pa_rate REAL,
    pitcher_hr_vuln REAL,
    park_factor REAL,
    platoon_hr_rate REAL,
    power_trend REAL,
    p_hr REAL,
    PRIMARY KEY (feature_date, batter_id)
);
"""

PARK_FACTORS_DDL = """
CREATE TABLE IF NOT EXISTS park_factors (
    team TEXT PRIMARY KEY,
    hr_park_factor REAL,
    sample_size INTEGER
);
"""

HR_FEATURES_V2_DDL = """
CREATE TABLE IF NOT EXISTS hr_features_v2 (
    feature_date TEXT,
    batter_id INTEGER,
    barrel_rate REAL,
    avg_exit_velo REAL,
    hr_pa_rate REAL,
    pitcher_hr_vuln REAL,
    park_factor REAL,
    platoon_hr_rate REAL,
    power_trend REAL,
    fly_ball_rate REAL,
    hard_hit_fly_rate REAL,
    pitcher_recent_hr_vuln REAL,
    p_hr_v2 REAL,
    PRIMARY KEY (feature_date, batter_id)
);
"""

HR_LEADERBOARD_DDL = """
CREATE TABLE IF NOT EXISTS hr_leaderboard (
    prediction_date TEXT,
    player_id INTEGER,
    player_name TEXT,
    team TEXT,
    hr_rank INTEGER,
    hr_score REAL,
    p_hr REAL,
    barrel_rate REAL,
    fly_ball_rate REAL,
    hard_hit_fly_rate REAL,
    park_factor REAL,
    active_hr_signal_count INTEGER,
    top_hr_signal TEXT,
    PRIMARY KEY (prediction_date, player_id)
);
"""

HR_PREDICTION_TRACKING_DDL = """
CREATE TABLE IF NOT EXISTS hr_prediction_tracking (
    prediction_date TEXT,
    player_id INTEGER,
    hr_rank INTEGER,
    hr_score REAL,
    p_hr REAL,
    actual_hr INTEGER,
    hr_correct INTEGER,
    PRIMARY KEY (prediction_date, player_id)
);
"""

HR_RETRO_ANALYSIS_DDL = """
CREATE TABLE IF NOT EXISTS hr_retro_analysis (
    analysis_date    TEXT,
    analysis_type    TEXT,
    signal_type      TEXT NOT NULL DEFAULT '_',
    metric_name      TEXT,
    metric_value     REAL,
    sample_size      INTEGER,
    detail_json      TEXT,
    narrative        TEXT,
    PRIMARY KEY (analysis_date, analysis_type, signal_type, metric_name)
);
"""

BATTER_VS_TEAM_DDL = """
CREATE TABLE IF NOT EXISTS batter_vs_team (
    batter_id     INTEGER,
    opp_team      TEXT,
    games         INTEGER,
    pa            INTEGER,
    ab            INTEGER,
    hits          INTEGER,
    singles       INTEGER,
    doubles       INTEGER,
    triples       INTEGER,
    hr            INTEGER,
    bb            INTEGER,
    so            INTEGER,
    avg           REAL,
    obp           REAL,
    slg           REAL,
    last_updated  TEXT,
    PRIMARY KEY (batter_id, opp_team)
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
        conn.executescript(HR_FEATURES_DDL)
        conn.executescript(HR_FEATURES_V2_DDL)
        conn.executescript(PARK_FACTORS_DDL)
        conn.executescript(BATTER_VS_TEAM_DDL)
        conn.executescript(HR_LEADERBOARD_DDL)
        conn.executescript(HR_PREDICTION_TRACKING_DDL)
        conn.executescript(HR_RETRO_ANALYSIS_DDL)
        # Schema migrations: add columns to existing tables (idempotent)
        for stmt in [
            "ALTER TABLE statcast_staging ADD COLUMN pitch_type TEXT",
            "ALTER TABLE statcast_staging ADD COLUMN release_speed REAL",
        ]:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # Column already exists
    logger.info("Phase 3 tables verified.")
