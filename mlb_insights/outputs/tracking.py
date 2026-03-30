"""
mlb_insights/outputs/tracking.py -- Score predictions against actuals.

Functions:
    score_yesterday(date, conn)             -- join yesterday's predictions with actuals
    update_calibration_summary(date, conn)  -- compute rolling Brier scores
"""

import logging
import sqlite3
from datetime import datetime, timedelta

import numpy as np

from mlb_insights.config import BRIER_ROLLING_WINDOWS
from mlb_insights.signal_engine.calibration import brier_score, brier_skill_score

logger = logging.getLogger(__name__)


def score_yesterday(date_str: str, conn: sqlite3.Connection) -> int:
    """Score yesterday's predictions against actual batter_stats results.

    Looks up prediction_tracking rows for yesterday (date - 1 day), joins
    with batter_stats actuals, and updates the actual_hits, actual_hr,
    actual_pa, and correctness columns.

    Args:
        date_str: Today's date (YYYY-MM-DD). Yesterday = date - 1.
        conn: Open sqlite3 connection.

    Returns:
        Number of predictions scored.
    """
    yesterday = (
        datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    # Get yesterday's predictions that still lack actuals
    preds = conn.execute("""
        SELECT prediction_date, player_id, p_1hit, p_2hit, p_hr
        FROM prediction_tracking
        WHERE prediction_date = ? AND actual_hits IS NULL
    """, (yesterday,)).fetchall()

    if not preds:
        logger.info("No unscored predictions for %s.", yesterday)
        return 0

    # Get actual batter stats for yesterday
    actuals = {}
    rows = conn.execute("""
        SELECT batter_id, hits, hr, pa
        FROM batter_stats
        WHERE game_date = ?
    """, (yesterday,)).fetchall()
    for r in rows:
        actuals[r["batter_id"]] = {
            "hits": r["hits"],
            "hr": r["hr"],
            "pa": r["pa"],
        }

    scored = 0
    for pred in preds:
        pid = pred["player_id"]
        if pid not in actuals:
            continue

        actual = actuals[pid]
        hits = actual["hits"]
        hr = actual["hr"] or 0
        pa = actual["pa"]

        hit_1_correct = 1 if hits >= 1 else 0
        hit_2_correct = 1 if hits >= 2 else 0
        hr_correct = 1 if hr >= 1 else 0

        conn.execute("""
            UPDATE prediction_tracking
            SET actual_hits = ?, actual_hr = ?, actual_pa = ?,
                hit_1_correct = ?, hit_2_correct = ?, hr_correct = ?
            WHERE prediction_date = ? AND player_id = ?
        """, (
            hits, hr, pa,
            hit_1_correct, hit_2_correct, hr_correct,
            yesterday, pid,
        ))
        scored += 1

    conn.commit()
    logger.info("Scored %d/%d predictions for %s.", scored, len(preds), yesterday)
    return scored


def score_hr_watch(date_str: str, conn: sqlite3.Connection) -> int:
    """Score yesterday's HR Watch predictions against actuals.

    Checks the top 10 HR Watch candidates (by p_hr from hr_features)
    and counts how many actually hit HRs.

    Args:
        date_str: Today's date (YYYY-MM-DD). Yesterday = date - 1.
        conn: Open sqlite3 connection.

    Returns:
        Number of HR Watch hits in top 10.
    """
    yesterday = (
        datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    # Get top 10 HR candidates from hr_features
    top_hr = conn.execute("""
        SELECT batter_id, p_hr
        FROM hr_features
        WHERE feature_date = ?
        ORDER BY p_hr DESC
        LIMIT 10
    """, (yesterday,)).fetchall()

    if not top_hr:
        return 0

    # Get actuals
    hits = 0
    total = len(top_hr)
    for row in top_hr:
        actual = conn.execute("""
            SELECT hr FROM batter_stats
            WHERE batter_id = ? AND game_date = ?
        """, (row["batter_id"], yesterday)).fetchone()

        if actual and (actual["hr"] or 0) >= 1:
            hits += 1

    # Store in calibration_summary
    if total > 0:
        accuracy = hits / total
        conn.execute("""
            INSERT OR REPLACE INTO calibration_summary
                (summary_date, metric_type, window_days, value, sample_size)
            VALUES (?, 'hr_watch_top10_accuracy', 1, ?, ?)
        """, (date_str, accuracy, total))
        conn.commit()

        logger.info(
            "HR Watch top-10 accuracy for %s: %d/%d (%.1f%%)",
            yesterday, hits, total, accuracy * 100,
        )

    return hits


def update_calibration_summary(date_str: str, conn: sqlite3.Connection):
    """Compute rolling Brier scores and save to calibration_summary.

    For each window in BRIER_ROLLING_WINDOWS, computes Brier score and
    BSS for 1-hit, 2-hit, and HR predictions.

    Args:
        date_str: Current date.
        conn: Open sqlite3 connection.
    """
    for window in BRIER_ROLLING_WINDOWS:
        start_date = (
            datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=window)
        ).strftime("%Y-%m-%d")

        rows = conn.execute("""
            SELECT p_1hit, p_2hit, p_hr, actual_hits, actual_hr
            FROM prediction_tracking
            WHERE prediction_date > ? AND prediction_date <= ?
              AND actual_hits IS NOT NULL
        """, (start_date, date_str)).fetchall()

        if len(rows) < 10:
            continue

        p1 = np.array([r["p_1hit"] for r in rows])
        y1 = np.array([1.0 if r["actual_hits"] >= 1 else 0.0 for r in rows])
        p2 = np.array([r["p_2hit"] for r in rows])
        y2 = np.array([1.0 if r["actual_hits"] >= 2 else 0.0 for r in rows])
        phr = np.array([r["p_hr"] for r in rows])
        yhr = np.array([1.0 if (r["actual_hr"] or 0) >= 1 else 0.0 for r in rows])

        metrics = [
            ("brier_1hit", brier_score(p1, y1)),
            ("brier_2hit", brier_score(p2, y2)),
            ("brier_hr", brier_score(phr, yhr)),
            ("bss_1hit", brier_skill_score(p1, y1)),
            ("bss_2hit", brier_skill_score(p2, y2)),
            ("bss_hr", brier_skill_score(phr, yhr)),
        ]

        for metric_type, value in metrics:
            conn.execute("""
                INSERT OR REPLACE INTO calibration_summary
                    (summary_date, metric_type, window_days, value, sample_size)
                VALUES (?,?,?,?,?)
            """, (date_str, metric_type, window, value, len(rows)))

    conn.commit()
    logger.info("Updated calibration summary for %s.", date_str)


def print_tracking_summary(date_str: str, conn: sqlite3.Connection):
    """Print a summary of recent prediction accuracy to stdout.

    Args:
        date_str: Current date.
        conn: Open sqlite3 connection.
    """
    yesterday = (
        datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    # Yesterday's results
    rows = conn.execute("""
        SELECT daily_rank, p_1hit, actual_hits, actual_hr
        FROM prediction_tracking
        WHERE prediction_date = ? AND actual_hits IS NOT NULL
        ORDER BY daily_rank
    """, (yesterday,)).fetchall()

    if not rows:
        print(f"  No scored predictions for {yesterday}.")
        return

    total = len(rows)
    hit_1_rate = sum(1 for r in rows if r["actual_hits"] >= 1) / total
    hit_2_rate = sum(1 for r in rows if r["actual_hits"] >= 2) / total

    # Top 10 performance
    top10 = [r for r in rows if r["daily_rank"] <= 10]
    top10_1hit = sum(1 for r in top10 if r["actual_hits"] >= 1) / len(top10) if top10 else 0

    # Top 25 performance
    top25 = [r for r in rows if r["daily_rank"] <= 25]
    top25_1hit = sum(1 for r in top25 if r["actual_hits"] >= 1) / len(top25) if top25 else 0

    print(f"\n  Tracking Summary for {yesterday}:")
    print(f"    Total scored: {total}")
    print(f"    Overall 1+ hit rate: {hit_1_rate:.3f}")
    print(f"    Overall 2+ hit rate: {hit_2_rate:.3f}")
    if top10:
        print(f"    Top 10 hit rate: {top10_1hit:.3f} ({sum(1 for r in top10 if r['actual_hits'] >= 1)}/{len(top10)})")
    if top25:
        print(f"    Top 25 hit rate: {top25_1hit:.3f} ({sum(1 for r in top25 if r['actual_hits'] >= 1)}/{len(top25)})")

    # Rolling calibration summary
    cal = conn.execute("""
        SELECT metric_type, window_days, value, sample_size
        FROM calibration_summary
        WHERE summary_date = ?
        ORDER BY window_days, metric_type
    """, (date_str,)).fetchall()

    if cal:
        print(f"\n  Calibration Summary (as of {date_str}):")
        for r in cal:
            print(f"    {r['metric_type']} ({r['window_days']}d): {r['value']:.6f} (n={r['sample_size']})")
