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
    ]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()
    df = df.dropna(subset=["game_pk", "batter", "pitcher"])

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
    """Fetch today's games and probable pitchers from the schedules table.

    If schedules are not yet populated for this date, attempt to pull
    via the SportsDataIO client (shared/sdio_client.py).  Falls back
    gracefully if the API is unavailable.

    Args:
        date_str: Date in YYYY-MM-DD format.
        conn: Open sqlite3 connection.

    Returns:
        List of dicts with keys: game_id, home_team, away_team,
        home_pitcher_id, away_pitcher_id, lineup_confirmed.
    """
    # Check existing schedule data
    rows = conn.execute(
        "SELECT * FROM schedules WHERE game_date = ?", (date_str,)
    ).fetchall()

    if not rows:
        logger.info("No schedule data for %s. Attempting SDIO pull ...", date_str)
        try:
            _pull_schedule_from_sdio(date_str, conn)
            rows = conn.execute(
                "SELECT * FROM schedules WHERE game_date = ?", (date_str,)
            ).fetchall()
        except Exception as exc:
            logger.warning("Schedule pull failed for %s: %s", date_str, exc)
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


def _pull_schedule_from_sdio(date_str: str, conn: sqlite3.Connection):
    """Internal: pull schedule from SDIO API and write to schedules table."""
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
