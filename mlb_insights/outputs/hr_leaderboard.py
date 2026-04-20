"""
mlb_insights/outputs/hr_leaderboard.py -- HR leaderboard and prediction tracking.

Writes HR-specific leaderboard rows and prediction tracking for the
HR prediction pipeline.  Follows the same patterns as leaderboard.py.
"""

import logging
import sqlite3

from mlb_insights.signal_engine.hr_composite import HRScoredPlayer

logger = logging.getLogger(__name__)


def write_hr_leaderboard(
    conn: sqlite3.Connection,
    date_str: str,
    hr_players: list[HRScoredPlayer],
    top_n: int = 15,
):
    """Write hr_leaderboard rows for a single date.

    Deletes existing rows for the date, then inserts the top N.

    Args:
        conn: Open sqlite3 connection.
        date_str: Prediction date.
        hr_players: Ranked HR player list.
        top_n: Number of players to write.
    """
    top = [p for p in hr_players if p.date == date_str][:top_n]
    if not top:
        logger.warning("No HR players to write to leaderboard for %s.", date_str)
        return

    conn.execute(
        "DELETE FROM hr_leaderboard WHERE prediction_date = ?", (date_str,)
    )

    # Build name lookup from player_lookup table
    name_map = {}
    try:
        for row in conn.execute("SELECT mlbam_id, player_name FROM player_lookup"):
            name_map[row[0]] = row[1]
    except Exception:
        pass

    # Build team lookup from latest batter_stats
    team_map = {}
    try:
        for row in conn.execute(
            "SELECT batter_id, team FROM batter_stats "
            "WHERE game_date = ? AND team IS NOT NULL", (date_str,)
        ):
            team_map[row[0]] = row[1]
    except Exception:
        pass

    rows = []
    for p in top:
        rows.append((
            date_str,
            p.player_id,
            name_map.get(p.player_id),
            team_map.get(p.player_id),
            p.hr_rank,
            p.hr_score,
            p.p_hr,
            p.barrel_rate,
            p.fly_ball_rate,
            p.hard_hit_fly_rate,
            p.park_factor,
            p.active_hr_signal_count,
            p.top_hr_signal,
        ))

    conn.executemany("""
        INSERT INTO hr_leaderboard
            (prediction_date, player_id, player_name, team,
             hr_rank, hr_score, p_hr,
             barrel_rate, fly_ball_rate, hard_hit_fly_rate, park_factor,
             active_hr_signal_count, top_hr_signal)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    logger.info("Wrote %d HR leaderboard rows for %s.", len(rows), date_str)


def write_hr_prediction_tracking(
    conn: sqlite3.Connection,
    date_str: str,
    hr_players: list[HRScoredPlayer],
):
    """Write hr_prediction_tracking rows for a single date.

    Actuals are left NULL -- they get filled in by tracking.

    Args:
        conn: Open sqlite3 connection.
        date_str: Prediction date.
        hr_players: Ranked HR players.
    """
    conn.execute(
        "DELETE FROM hr_prediction_tracking WHERE prediction_date = ?", (date_str,)
    )

    rows = []
    for p in hr_players:
        if p.date != date_str:
            continue
        rows.append((
            date_str,
            p.player_id,
            p.hr_rank,
            p.hr_score,
            p.p_hr,
            p.actual_hr,
            (1 if (p.actual_hr or 0) >= 1 else 0) if p.actual_hr is not None else None,
        ))

    conn.executemany("""
        INSERT OR REPLACE INTO hr_prediction_tracking
            (prediction_date, player_id, hr_rank, hr_score, p_hr,
             actual_hr, hr_correct)
        VALUES (?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    logger.info("Wrote %d HR prediction_tracking rows for %s.", len(rows), date_str)


def print_hr_leaderboard(
    hr_players: list[HRScoredPlayer],
    date_str: str,
    top_n: int = 15,
):
    """Print a formatted HR leaderboard summary to stdout.

    Args:
        hr_players: Ranked HR players.
        date_str: Display date.
        top_n: Number of players to display.
    """
    top = [p for p in hr_players if p.date == date_str][:top_n]
    if not top:
        print(f"  No HR leaderboard data for {date_str}.")
        return

    print(f"\n{'='*80}")
    print(f"  HR INSIGHTS LEADERBOARD -- {date_str}")
    print(f"{'='*80}")
    print(
        f"  {'Rank':>4}  {'Player':>10}  {'P(HR)':>6}  {'Score':>6}  "
        f"{'Barrel':>7}  {'FB%':>5}  {'Park':>5}  {'Top Signal'}"
    )
    print(
        f"  {'-'*4}  {'-'*10}  {'-'*6}  {'-'*6}  "
        f"{'-'*7}  {'-'*5}  {'-'*5}  {'-'*20}"
    )

    for p in top:
        sig_label = p.top_hr_signal or "-"
        print(
            f"  {p.hr_rank:>4}  {p.player_id:>10}  {p.p_hr:>6.3f}  {p.hr_score:>6.1f}  "
            f"{p.barrel_rate:>7.3f}  {p.fly_ball_rate:>5.3f}  {p.park_factor:>5.2f}  "
            f"{sig_label}"
        )


def format_hr_telegram(
    hr_players: list[HRScoredPlayer],
    date_str: str,
    top_n: int = 10,
) -> str:
    """Format HR leaderboard for Telegram output.

    Args:
        hr_players: Ranked HR players.
        date_str: Display date.
        top_n: Number of players to include.

    Returns:
        Formatted string suitable for Telegram.
    """
    top = [p for p in hr_players if p.date == date_str][:top_n]
    if not top:
        return f"No HR leaderboard data for {date_str}."

    lines = [
        f"<b>HR Leaderboard -- {date_str}</b>",
        "",
    ]

    for p in top:
        sig_label = p.top_hr_signal or "-"
        lines.append(
            f"{p.hr_rank}. <b>{p.player_id}</b> | "
            f"P(HR) {p.p_hr:.3f} | Score {p.hr_score:.1f} | "
            f"Barrel {p.barrel_rate:.3f} | FB% {p.fly_ball_rate:.3f} | "
            f"Park {p.park_factor:.2f} | {sig_label}"
        )

    return "\n".join(lines)
