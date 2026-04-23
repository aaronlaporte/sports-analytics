"""
mlb_insights/data_pipeline/ingest.py -- Pull Statcast data and game schedules.

Functions:
    pull_statcast(date, conn)  -- fetch one day of statcast and upsert
    pull_schedule(date, conn)  -- fetch today's games and probable pitchers
"""

import logging
import sqlite3
import warnings
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Suppress pybaseball FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")


# ── Statcast ─────────────────────────────────────────────────────────────────

def _date_has_statcast(conn: sqlite3.Connection, date_str: str) -> bool:
    """Check whether statcast_staging already has rows for this date."""
    row = conn.execute(
        "SELECT COUNT(*) as n FROM statcast_staging WHERE game_date = ?",
        (date_str,),
    ).fetchone()
    return (row["n"] or 0) > 0


def pull_statcast(date_str: str, conn: sqlite3.Connection) -> int:
    """Pull one day of Statcast data via pybaseball and upsert into statcast_staging.

    Args:
        date_str: Date in YYYY-MM-DD format.
        conn: Open sqlite3 connection.

    Returns:
        Number of rows upserted (0 if data was already present or unavailable).
    """
    if _date_has_statcast(conn, date_str):
        logger.info("Statcast data already present for %s, skipping.", date_str)
        return 0

    try:
        from pybaseball import statcast
    except ImportError:
        logger.error("pybaseball not installed. Run: pip install pybaseball")
        return 0

    logger.info("Pulling Statcast for %s ...", date_str)
    try:
        df = statcast(date_str, date_str)
    except Exception as exc:
        logger.warning("Statcast pull failed for %s: %s", date_str, exc)
        return 0

    if df is None or df.empty:
        logger.info("No Statcast data for %s (off day or games not complete).", date_str)
        return 0

    # Ensure game_date is string
    df["game_date"] = df["game_date"].astype(str).str[:10]

    # Select columns that match statcast_staging schema
    keep_cols = [
        "game_pk", "game_date", "batter", "pitcher",
        "events", "description", "bb_type",
        "hit_distance_sc", "launch_speed", "launch_angle",
        "estimated_ba_using_speedangle",
        "stand", "p_throws", "home_team", "away_team",
        "inning_topbot", "at_bat_number", "pitch_number",
        "day_night", "pitch_type", "release_speed",
    ]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()
    df = df.dropna(subset=["game_pk", "batter", "pitcher"])

    # Convert pandas NA/NaN to None for SQLite compatibility
    df = df.astype(object).where(df.notna(), None)

    rows = list(df.itertuples(index=False, name=None))
    placeholders = ",".join(["?"] * len(available))
    col_names = ",".join(available)

    conn.executemany(
        f"INSERT OR IGNORE INTO statcast_staging ({col_names}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    logger.info("Upserted %d statcast rows for %s.", len(rows), date_str)
    return len(rows)


def pull_statcast_range(start: str, end: str, conn: sqlite3.Connection) -> int:
    """Pull Statcast for a date range, skipping dates already present.

    Args:
        start: Start date YYYY-MM-DD.
        end: End date YYYY-MM-DD.
        conn: Open sqlite3 connection.

    Returns:
        Total rows upserted.
    """
    import pandas as pd

    total = 0
    dates = pd.date_range(start, end, freq="D")
    for dt in dates:
        ds = dt.strftime("%Y-%m-%d")
        total += pull_statcast(ds, conn)
    return total


# ── Schedules ────────────────────────────────────────────────────────────────

def pull_schedule(date_str: str, conn: sqlite3.Connection) -> list[dict]:
    """Fetch games for a date from the schedules table.

    Attempts three sources in order:
      1. Existing rows in the ``schedules`` table.
      2. MLB Stats API (free, no key required).
      3. SportsDataIO API (requires API key — used for lineup detail).

    Args:
        date_str: Date in YYYY-MM-DD format.
        conn: Open sqlite3 connection.

    Returns:
        List of dicts with keys: game_id, home_team, away_team,
        home_pitcher_id, away_pitcher_id, lineup_confirmed, status.
    """
    # Check existing schedule data
    rows = conn.execute(
        "SELECT * FROM schedules WHERE game_date = ?", (date_str,)
    ).fetchall()

    if not rows:
        # Try free MLB Stats API first (always available)
        try:
            _pull_schedule_from_mlb_api(date_str, conn)
            rows = conn.execute(
                "SELECT * FROM schedules WHERE game_date = ?", (date_str,)
            ).fetchall()
        except Exception as exc:
            logger.warning("MLB Stats API schedule pull failed for %s: %s", date_str, exc)

    if not rows:
        # Fallback: try SDIO (requires API key)
        logger.info("No schedule from MLB API for %s. Trying SDIO ...", date_str)
        try:
            _pull_schedule_from_sdio(date_str, conn)
            rows = conn.execute(
                "SELECT * FROM schedules WHERE game_date = ?", (date_str,)
            ).fetchall()
        except Exception as exc:
            logger.warning("SDIO schedule pull also failed for %s: %s", date_str, exc)
            return []

    games = []
    for r in rows:
        games.append({
            "game_id": r["game_id"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "home_pitcher_id": r["home_pitcher_id"],
            "away_pitcher_id": r["away_pitcher_id"],
            "lineup_confirmed": r["lineup_confirmed"],
            "status": r["status"],
        })

    logger.info("Schedule for %s: %d games loaded.", date_str, len(games))
    return games


def _pull_schedule_from_mlb_api(date_str: str, conn: sqlite3.Connection):
    """Pull schedule from the free MLB Stats API (statsapi.mlb.com).

    No API key required.  Populates the ``schedules`` table with game_id,
    teams, and status.  Does not include pitcher IDs (use SDIO for that).
    """
    import requests

    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?date={date_str}&sportId=1&hydrate=team"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    dates = data.get("dates", [])
    if not dates:
        logger.info("MLB Stats API returned no dates for %s.", date_str)
        return

    games = dates[0].get("games", [])
    if not games:
        logger.info("MLB Stats API: no games on %s.", date_str)
        return

    inserted = 0
    for g in games:
        game_pk = g.get("gamePk")
        if not game_pk:
            continue
        # Only include regular season / postseason games
        game_type = g.get("gameType", "")
        if game_type not in ("R", "F", "D", "L", "W", "P"):
            continue

        home_team = g.get("teams", {}).get("home", {}).get("team", {})
        away_team = g.get("teams", {}).get("away", {}).get("team", {})
        home_abbr = home_team.get("abbreviation")
        away_abbr = away_team.get("abbreviation")
        status = g.get("status", {}).get("detailedState", "Scheduled")

        # Skip postponed / cancelled
        if status in ("Postponed", "Cancelled", "Suspended"):
            continue

        conn.execute("""
            INSERT OR IGNORE INTO schedules
                (game_id, game_date, home_team, away_team, status,
                 lineup_confirmed, home_pitcher_id, away_pitcher_id)
            VALUES (?,?,?,?,?,0,NULL,NULL)
        """, (game_pk, date_str, home_abbr, away_abbr, status))
        inserted += 1

    conn.commit()
    logger.info(
        "MLB Stats API: %d games inserted for %s.", inserted, date_str,
    )


def _pull_schedule_from_sdio(date_str: str, conn: sqlite3.Connection):
    """Pull schedule from SDIO API and write to schedules table.

    Requires an API key (SDIO_API_KEY env var or keys/sdio_key.txt).
    Provides lineup confirmation and starting pitcher IDs that the
    free MLB Stats API does not include.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from shared.sdio_client import SDIOClient

    client = SDIOClient("mlb")
    games = client.mlb_games_by_date(date_str)
    if not games:
        logger.info("SDIO returned no games for %s.", date_str)
        return

    for g in games:
        game_id = g.get("GameID")
        if not game_id:
            continue
        lineup_confirmed = 1 if g.get("LineupConfirmed") else 0
        home_pitcher_id = None
        away_pitcher_id = None

        if lineup_confirmed:
            try:
                lineup = client.mlb_lineups(game_id)
                if isinstance(lineup, dict):
                    home_pitcher_id = lineup.get("HomeStartingPitcherID")
                    away_pitcher_id = lineup.get("AwayStartingPitcherID")
            except Exception:
                pass

        conn.execute("""
            INSERT OR REPLACE INTO schedules
                (game_id, game_date, home_team, away_team, status,
                 lineup_confirmed, home_pitcher_id, away_pitcher_id)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            game_id,
            g.get("Day", date_str)[:10],
            g.get("HomeTeam"),
            g.get("AwayTeam"),
            g.get("Status"),
            lineup_confirmed,
            home_pitcher_id,
            away_pitcher_id,
        ))

    conn.commit()
    logger.info("SDIO schedule: %d games upserted for %s.", len(games), date_str)
