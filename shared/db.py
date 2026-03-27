"""
shared/db.py — SQLite connection helpers and schema definitions.

Each sport has its own database file. Call create_schema(sport) once
to initialize tables, then use get_conn(sport) for all queries.
"""

import sqlite3
from pathlib import Path
from shared.config import MLB_DB, NBA_DB, NFL_DB


def get_conn(sport: str) -> sqlite3.Connection:
    """Return a connection for the given sport ('mlb', 'nba', 'nfl')."""
    db_path = {"mlb": MLB_DB, "nba": NBA_DB, "nfl": NFL_DB}[sport.lower()]
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── MLB Schema ────────────────────────────────────────────────────────────────

MLB_SCHEMA = """
CREATE TABLE IF NOT EXISTS statcast_raw (
    game_pk       INTEGER,
    game_date     TEXT,
    batter        INTEGER,
    pitcher       INTEGER,
    events        TEXT,
    description   TEXT,
    bb_type       TEXT,
    hit_distance_sc REAL,
    launch_speed  REAL,
    launch_angle  REAL,
    estimated_ba_using_speedangle REAL,
    PRIMARY KEY (game_pk, batter, pitcher, description)
);

CREATE TABLE IF NOT EXISTS batter_stats (
    game_date     TEXT,
    batter_id     INTEGER,
    batter_name   TEXT,
    team          TEXT,
    pa            INTEGER,
    ab            INTEGER,
    hits          INTEGER,
    hr            INTEGER,
    bb            INTEGER,
    so            INTEGER,
    avg           REAL,
    obp           REAL,
    slg           REAL,
    -- rolling features
    hits_pg_1     REAL, hits_pg_2  REAL, hits_pg_3  REAL,
    hits_pg_5     REAL, hits_pg_10 REAL, hits_pg_20 REAL,
    current_streak INTEGER,
    rebound_tier  TEXT,
    -- season aggregates
    season_hits   INTEGER,
    season_pa     INTEGER,
    season_hit_pct REAL,
    PRIMARY KEY (game_date, batter_id)
);

CREATE TABLE IF NOT EXISTS pitcher_stats (
    game_date     TEXT,
    pitcher_id    INTEGER,
    pitcher_name  TEXT,
    team          TEXT,
    throws        TEXT,  -- L or R
    batters_faced INTEGER,
    hits_allowed  INTEGER,
    hr_allowed    INTEGER,
    bb_allowed    INTEGER,
    so            INTEGER,
    avg_against   REAL,
    PRIMARY KEY (game_date, pitcher_id)
);

CREATE TABLE IF NOT EXISTS matchup_stats (
    batter_id     INTEGER,
    pitcher_id    INTEGER,
    pa            INTEGER,
    hits          INTEGER,
    hr            INTEGER,
    bb            INTEGER,
    so            INTEGER,
    avg           REAL,
    obp           REAL,
    slg           REAL,
    last_updated  TEXT,
    PRIMARY KEY (batter_id, pitcher_id)
);

CREATE TABLE IF NOT EXISTS schedules (
    game_id       INTEGER PRIMARY KEY,
    game_date     TEXT,
    home_team     TEXT,
    away_team     TEXT,
    status        TEXT,
    lineup_confirmed INTEGER DEFAULT 0,
    home_pitcher_id  INTEGER,
    away_pitcher_id  INTEGER
);

CREATE TABLE IF NOT EXISTS model_params (
    run_date      TEXT PRIMARY KEY,
    league_hit_per_pa REAL,
    season_prior_pa   INTEGER,
    recent_weight     REAL,
    split_weight      REAL,
    brier_score       REAL,
    params_json   TEXT
);

CREATE TABLE IF NOT EXISTS daily_picks (
    pick_date     TEXT,
    batter_id     INTEGER,
    batter_name   TEXT,
    team          TEXT,
    opponent      TEXT,
    pitcher_name  TEXT,
    pitcher_throws TEXT,
    model_prob    REAL,
    expected_pa   REAL,
    p_hit         REAL,
    odds_over     INTEGER,
    line          REAL,
    ev            REAL,
    value_bet     INTEGER DEFAULT 0,
    PRIMARY KEY (pick_date, batter_id)
);

CREATE TABLE IF NOT EXISTS live_odds (
    fetched_at    TEXT,
    event_id      TEXT,
    home_team     TEXT,
    away_team     TEXT,
    bookmaker     TEXT,
    market        TEXT,
    player        TEXT,
    name          TEXT,
    line          REAL,
    price         INTEGER,
    PRIMARY KEY (fetched_at, event_id, bookmaker, market, player, name)
);

CREATE TABLE IF NOT EXISTS bets (
    bet_date      TEXT,
    sport         TEXT DEFAULT 'MLB',
    player        TEXT,
    bet_type      TEXT,
    line          REAL,
    odds          INTEGER,
    stake         REAL,
    result        TEXT,  -- Win, Loss, Push, Pending
    pnl           REAL,
    notes         TEXT
);

-- ── MLB Insights Platform tables ────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS daily_signals (
    signal_date   TEXT,
    player_id     INTEGER,
    signal_type   TEXT,
    confidence    REAL,
    headline      TEXT,
    reasons_json  TEXT,
    interpretation TEXT,
    PRIMARY KEY (signal_date, player_id, signal_type)
);

CREATE TABLE IF NOT EXISTS daily_leaderboard (
    prediction_date TEXT,
    player_id       INTEGER,
    player_name     TEXT,
    team            TEXT,
    opponent        TEXT,
    opp_pitcher     TEXT,
    daily_rank      INTEGER,
    daily_score     REAL,
    p_1hit          REAL,
    p_2hit          REAL,
    p_hr            REAL,
    active_signal_count INTEGER,
    top_signal      TEXT,
    top_reason      TEXT,
    PRIMARY KEY (prediction_date, player_id)
);

CREATE TABLE IF NOT EXISTS prediction_tracking (
    prediction_date TEXT,
    player_id       INTEGER,
    player_name     TEXT,
    daily_rank      INTEGER,
    daily_score     REAL,
    p_1hit          REAL,
    p_2hit          REAL,
    p_hr            REAL,
    actual_hits     INTEGER,
    actual_hr       INTEGER,
    actual_pa       INTEGER,
    hit_1_correct   INTEGER,
    hit_2_correct   INTEGER,
    hr_correct      INTEGER,
    PRIMARY KEY (prediction_date, player_id)
);

CREATE TABLE IF NOT EXISTS calibration_summary (
    summary_date    TEXT,
    metric_type     TEXT,
    window_days     INTEGER,
    value           REAL,
    sample_size     INTEGER,
    PRIMARY KEY (summary_date, metric_type, window_days)
);
"""

# ── NBA Schema ────────────────────────────────────────────────────────────────

NBA_SCHEMA = """
CREATE TABLE IF NOT EXISTS player_game_stats (
    stat_id         INTEGER PRIMARY KEY,
    player_id       INTEGER,
    player_name     TEXT,
    team            TEXT,
    opponent        TEXT,
    game_id         INTEGER,
    game_date       TEXT,
    season          INTEGER,
    season_type     INTEGER,
    home_or_away    TEXT,
    started         INTEGER,
    minutes         INTEGER,
    points          REAL,
    rebounds        REAL,
    assists         REAL,
    steals          REAL,
    blocks          REAL,
    turnovers       REAL,
    personal_fouls  REAL,
    three_pointers_made   REAL,
    three_pointers_att    REAL,
    field_goals_made      REAL,
    field_goals_att       REAL,
    field_goals_pct       REAL,
    free_throws_made      REAL,
    free_throws_att       REAL,
    off_rebounds    REAL,
    def_rebounds    REAL,
    usage_rate      REAL,
    true_shooting   REAL,
    player_efficiency REAL,
    plus_minus      REAL,
    fantasy_pts_dk  REAL,
    fantasy_pts_fd  REAL,
    salary_dk       INTEGER,
    salary_fd       INTEGER,
    salary_yahoo    INTEGER,
    opponent_rank   INTEGER,
    opp_position_rank INTEGER,
    injury_status   TEXT,
    lineup_confirmed INTEGER,
    is_game_over    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_pgs_player_date ON player_game_stats(player_id, game_date);
CREATE INDEX IF NOT EXISTS idx_pgs_date ON player_game_stats(game_date);

CREATE TABLE IF NOT EXISTS schedules (
    game_id       INTEGER PRIMARY KEY,
    game_date     TEXT,
    season        INTEGER,
    season_type   INTEGER,
    home_team     TEXT,
    away_team     TEXT,
    home_score    INTEGER,
    away_score    INTEGER,
    status        TEXT
);

CREATE TABLE IF NOT EXISTS daily_picks (
    pick_date     TEXT,
    player_id     INTEGER,
    player_name   TEXT,
    team          TEXT,
    opponent      TEXT,
    position      TEXT,
    prop_type     TEXT,    -- points, rebounds, assists, threes
    model_prob    REAL,
    direction     TEXT,    -- Over / Under
    line          REAL,
    best_odds     INTEGER,
    best_book     TEXT,
    ev            REAL,
    value_bet     INTEGER DEFAULT 0,
    PRIMARY KEY (pick_date, player_id, prop_type, direction)
);

CREATE TABLE IF NOT EXISTS live_odds (
    fetched_at    TEXT,
    event_id      TEXT,
    home_team     TEXT,
    away_team     TEXT,
    bookmaker     TEXT,
    market        TEXT,
    player        TEXT,
    name          TEXT,
    line          REAL,
    price         INTEGER,
    PRIMARY KEY (fetched_at, event_id, bookmaker, market, player, name)
);
"""

# ── NFL Schema ────────────────────────────────────────────────────────────────

NFL_SCHEMA = """
CREATE TABLE IF NOT EXISTS player_game_stats (
    stat_id               INTEGER PRIMARY KEY,
    player_id             INTEGER,
    player_name           TEXT,
    team                  TEXT,
    opponent              TEXT,
    game_key              TEXT,
    season                TEXT,
    season_type           INTEGER,
    week                  INTEGER,
    game_date             TEXT,
    home_or_away          TEXT,
    position              TEXT,
    rush_attempts         REAL,
    rush_yards            REAL,
    rush_tds              REAL,
    receiving_targets     REAL,
    receptions            REAL,
    receiving_yards       REAL,
    receiving_tds         REAL,
    total_tds             REAL,
    fantasy_pts_dk        REAL,
    fantasy_pts_fd        REAL,
    injury_status         TEXT
);

CREATE INDEX IF NOT EXISTS idx_nfl_pgs_player ON player_game_stats(player_id, game_date);

CREATE TABLE IF NOT EXISTS schedules (
    game_key      TEXT PRIMARY KEY,
    season        TEXT,
    week          INTEGER,
    game_date     TEXT,
    home_team     TEXT,
    away_team     TEXT,
    home_score    INTEGER,
    away_score    INTEGER,
    point_spread  REAL,
    over_under    REAL
);

CREATE TABLE IF NOT EXISTS daily_picks (
    pick_date     TEXT,
    player_id     INTEGER,
    player_name   TEXT,
    team          TEXT,
    opponent      TEXT,
    position      TEXT,
    prop_type     TEXT,
    model_prob    REAL,
    direction     TEXT,
    line          REAL,
    best_odds     INTEGER,
    best_book     TEXT,
    ev            REAL,
    value_bet     INTEGER DEFAULT 0,
    PRIMARY KEY (pick_date, player_id, prop_type)
);

CREATE TABLE IF NOT EXISTS live_odds (
    fetched_at    TEXT,
    event_id      TEXT,
    home_team     TEXT,
    away_team     TEXT,
    bookmaker     TEXT,
    market        TEXT,
    player        TEXT,
    name          TEXT,
    line          REAL,
    price         INTEGER,
    PRIMARY KEY (fetched_at, event_id, bookmaker, market, player, name)
);
"""

# ── Schema Init ───────────────────────────────────────────────────────────────

SCHEMAS = {"mlb": MLB_SCHEMA, "nba": NBA_SCHEMA, "nfl": NFL_SCHEMA}


def create_schema(sport: str):
    """Initialize all tables for the given sport. Safe to call repeatedly."""
    with get_conn(sport) as conn:
        conn.executescript(SCHEMAS[sport])
    print(f"[db] {sport.upper()} schema ready")


def init_all():
    """Initialize all three sport databases."""
    for sport in ("mlb", "nba", "nfl"):
        create_schema(sport)


if __name__ == "__main__":
    init_all()
