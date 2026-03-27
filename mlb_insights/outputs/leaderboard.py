"""
mlb_insights/outputs/leaderboard.py -- Generate and write daily_leaderboard rows.

Writes the top N ranked players per date to the daily_leaderboard table.
"""

import json
import logging
import sqlite3

from mlb_insights.config import LEADERBOARD_TOP_N
from mlb_insights.signal_engine.composite import ScoredPlayer

logger = logging.getLogger(__name__)


def generate_leaderboard(
    date: str,
    ranked_players: list[ScoredPlayer],
    top_n: int = LEADERBOARD_TOP_N,
) -> list[ScoredPlayer]:
    """Select top N players for the leaderboard.

    Args:
        date: Prediction date.
        ranked_players: Players sorted by daily_score descending.
        top_n: Number of players to include.

    Returns:
        Top N players (already ranked).
    """
    return [p for p in ranked_players if p.date == date][:top_n]


def write_leaderboard(
    conn: sqlite3.Connection,
    date: str,
    ranked_players: list[ScoredPlayer],
    top_n: int = LEADERBOARD_TOP_N,
):
    """Write daily_leaderboard rows for a single date.

    Deletes existing rows for the date, then inserts the top N.

    Args:
        conn: Open sqlite3 connection.
        date: Prediction date.
        ranked_players: Ranked player list.
        top_n: Number of players to write.
    """
    top = [p for p in ranked_players if p.date == date][:top_n]
    if not top:
        logger.warning("No players to write to leaderboard for %s.", date)
        return

    conn.execute(
        "DELETE FROM daily_leaderboard WHERE prediction_date = ?", (date,)
    )

    rows = []
    for p in top:
        rows.append((
            date,
            p.player_id,
            None,  # player_name (could enrich later)
            None,  # team
            None,  # opponent
            None,  # opp_pitcher
            p.daily_rank,
            p.daily_score,
            p.cal_p1hit,
            p.cal_p2hit,
            p.cal_phr,
            p.active_signal_count,
            p.top_signal,
            p.top_reason,
        ))

    conn.executemany("""
        INSERT INTO daily_leaderboard
            (prediction_date, player_id, player_name, team, opponent, opp_pitcher,
             daily_rank, daily_score, p_1hit, p_2hit, p_hr,
             active_signal_count, top_signal, top_reason)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    logger.info("Wrote %d leaderboard rows for %s.", len(rows), date)


def write_signals(
    conn: sqlite3.Connection,
    date: str,
    ranked_players: list[ScoredPlayer],
):
    """Write daily_signals rows for all players on a date.

    Args:
        conn: Open sqlite3 connection.
        date: Signal date.
        ranked_players: Ranked players with signals attached.
    """
    conn.execute(
        "DELETE FROM daily_signals WHERE signal_date = ?", (date,)
    )

    count = 0
    for p in ranked_players:
        if p.date != date:
            continue
        for sig in p.signals:
            conn.execute("""
                INSERT OR REPLACE INTO daily_signals
                    (signal_date, player_id, signal_type, confidence,
                     headline, reasons_json, interpretation)
                VALUES (?,?,?,?,?,?,?)
            """, sig.to_db_tuple(date, p.player_id))
            count += 1

    conn.commit()
    logger.info("Wrote %d signal rows for %s.", count, date)


def write_prediction_tracking(
    conn: sqlite3.Connection,
    date: str,
    ranked_players: list[ScoredPlayer],
):
    """Write prediction_tracking rows for a single date.

    Actuals are left NULL -- they get filled in by tracking.score_yesterday().

    Args:
        conn: Open sqlite3 connection.
        date: Prediction date.
        ranked_players: Ranked players.
    """
    conn.execute(
        "DELETE FROM prediction_tracking WHERE prediction_date = ?", (date,)
    )

    rows = []
    for p in ranked_players:
        if p.date != date:
            continue
        rows.append((
            date,
            p.player_id,
            None,  # player_name
            p.daily_rank,
            p.daily_score,
            p.cal_p1hit,
            p.cal_p2hit,
            p.cal_phr,
            p.actual_hits,
            p.actual_hr,
            p.actual_pa,
            (1 if p.actual_hits >= 1 else 0) if p.actual_hits is not None else None,
            (1 if p.actual_hits >= 2 else 0) if p.actual_hits is not None else None,
            (1 if (p.actual_hr or 0) >= 1 else 0) if p.actual_hits is not None else None,
        ))

    conn.executemany("""
        INSERT OR REPLACE INTO prediction_tracking
            (prediction_date, player_id, player_name, daily_rank, daily_score,
             p_1hit, p_2hit, p_hr,
             actual_hits, actual_hr, actual_pa,
             hit_1_correct, hit_2_correct, hr_correct)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    logger.info("Wrote %d prediction_tracking rows for %s.", len(rows), date)


def print_leaderboard(ranked_players: list[ScoredPlayer], date: str, top_n: int = 25):
    """Print a formatted leaderboard summary to stdout.

    Args:
        ranked_players: Ranked players.
        date: Display date.
        top_n: Number of players to display.
    """
    top = [p for p in ranked_players if p.date == date][:top_n]
    if not top:
        print(f"  No leaderboard data for {date}.")
        return

    print(f"\n{'='*72}")
    print(f"  MLB INSIGHTS LEADERBOARD — {date}")
    print(f"{'='*72}")
    print(f"  {'Rank':>4}  {'Player':>10}  {'Score':>6}  {'P(1H)':>6}  {'P(2H)':>6}  {'P(HR)':>6}  {'Sigs':>4}  {'Top Signal'}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*4}  {'-'*20}")

    for p in top:
        sig_label = p.top_signal or "-"
        print(
            f"  {p.daily_rank:>4}  {p.player_id:>10}  {p.daily_score:>6.1f}  "
            f"{p.cal_p1hit:>6.3f}  {p.cal_p2hit:>6.3f}  {p.cal_phr:>6.3f}  "
            f"{p.active_signal_count:>4}  {sig_label}"
        )
