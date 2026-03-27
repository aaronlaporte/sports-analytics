"""
mlb_insights/outputs/player_page.py -- Generate player page data.

Builds the player_pages table with current form, active signals,
recent predictions vs actuals, and composite score info.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta

from mlb_insights.utils.db import ensure_tables

logger = logging.getLogger(__name__)


def generate_player_page_data(
    conn: sqlite3.Connection,
    player_id: int,
    date: str,
) -> dict | None:
    """Build a complete player page data dict for one player on one date.

    Pulls from batter_stats, daily_leaderboard, daily_signals, and
    prediction_tracking to assemble all player page fields.

    Args:
        conn: Open sqlite3 connection.
        player_id: Batter player ID.
        date: Page date (YYYY-MM-DD).

    Returns:
        Dict with all player page fields, or None if insufficient data.
    """
    # Current form from batter_stats
    bs = conn.execute("""
        SELECT * FROM batter_stats
        WHERE batter_id = ? AND game_date <= ?
        ORDER BY game_date DESC LIMIT 1
    """, (player_id, date)).fetchone()

    if not bs:
        return None

    # Leaderboard entry for today
    lb = conn.execute("""
        SELECT * FROM daily_leaderboard
        WHERE player_id = ? AND prediction_date = ?
    """, (player_id, date)).fetchone()

    # Active signals for today
    signals = conn.execute("""
        SELECT signal_type, confidence, headline, reasons_json, interpretation
        FROM daily_signals
        WHERE player_id = ? AND signal_date = ?
    """, (player_id, date)).fetchall()

    active_signals = []
    for s in signals:
        active_signals.append({
            "signal_type": s["signal_type"],
            "confidence": s["confidence"],
            "headline": s["headline"],
            "reasons": json.loads(s["reasons_json"]) if s["reasons_json"] else [],
            "interpretation": s["interpretation"],
        })

    # Last 10 predictions with actuals
    preds = conn.execute("""
        SELECT prediction_date, daily_rank, daily_score,
               p_1hit, actual_hits, actual_pa
        FROM prediction_tracking
        WHERE player_id = ? AND prediction_date <= ?
        ORDER BY prediction_date DESC
        LIMIT 10
    """, (player_id, date)).fetchall()

    last_10 = []
    for p in preds:
        predicted_hit = 1 if (p["p_1hit"] or 0) >= 0.5 else 0
        actual = None
        if p["actual_hits"] is not None:
            actual = 1 if p["actual_hits"] >= 1 else 0
        last_10.append({
            "date": p["prediction_date"],
            "rank": p["daily_rank"],
            "score": p["daily_score"],
            "predicted": predicted_hit,
            "actual": actual,
        })

    # 30-day prediction accuracy
    date_30d_ago = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    acc_rows = conn.execute("""
        SELECT p_1hit, actual_hits
        FROM prediction_tracking
        WHERE player_id = ? AND prediction_date > ? AND prediction_date <= ?
          AND actual_hits IS NOT NULL
    """, (player_id, date_30d_ago, date)).fetchall()

    accuracy_30d = None
    if acc_rows:
        correct = sum(
            1 for r in acc_rows
            if (1 if r["p_1hit"] >= 0.5 else 0) == (1 if r["actual_hits"] >= 1 else 0)
        )
        accuracy_30d = round(correct / len(acc_rows), 3)

    return {
        "page_date": date,
        "player_id": player_id,
        "player_name": bs["batter_name"],
        "team": bs["team"],
        "season_avg": bs["avg"],
        "season_hit_pct": bs["season_hit_pct"],
        "current_streak": bs["current_streak"],
        "hits_last_5": bs["hits_pg_5"],
        "hits_last_10": bs["hits_pg_10"],
        "active_signals_json": json.dumps(active_signals),
        "last_10_predictions_json": json.dumps(last_10),
        "prediction_accuracy_30d": accuracy_30d,
        "daily_rank": lb["daily_rank"] if lb else None,
        "daily_score": lb["daily_score"] if lb else None,
    }


def write_player_pages(
    conn: sqlite3.Connection,
    date: str,
    player_ids: list[int] | None = None,
):
    """Generate and write player page data for all leaderboard players on a date.

    If player_ids is None, writes pages for all players on that day's leaderboard.

    Args:
        conn: Open sqlite3 connection.
        date: Page date.
        player_ids: Optional list of specific player IDs. If None, uses
            all players from the daily_leaderboard for that date.
    """
    ensure_tables()

    if player_ids is None:
        rows = conn.execute(
            "SELECT player_id FROM daily_leaderboard WHERE prediction_date = ?",
            (date,),
        ).fetchall()
        player_ids = [r["player_id"] for r in rows]

    if not player_ids:
        logger.info("No players for player pages on %s.", date)
        return

    # Delete existing pages for this date
    conn.execute("DELETE FROM player_pages WHERE page_date = ?", (date,))

    written = 0
    for pid in player_ids:
        data = generate_player_page_data(conn, pid, date)
        if data is None:
            continue

        conn.execute("""
            INSERT OR REPLACE INTO player_pages
                (page_date, player_id, player_name, team,
                 season_avg, season_hit_pct, current_streak,
                 hits_last_5, hits_last_10,
                 active_signals_json,
                 last_10_predictions_json, prediction_accuracy_30d,
                 daily_rank, daily_score)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data["page_date"],
            data["player_id"],
            data["player_name"],
            data["team"],
            data["season_avg"],
            data["season_hit_pct"],
            data["current_streak"],
            data["hits_last_5"],
            data["hits_last_10"],
            data["active_signals_json"],
            data["last_10_predictions_json"],
            data["prediction_accuracy_30d"],
            data["daily_rank"],
            data["daily_score"],
        ))
        written += 1

    conn.commit()
    logger.info("Wrote %d player page rows for %s.", written, date)
