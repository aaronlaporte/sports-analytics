"""
mlb_insights/outputs/hr_tracking.py -- Score HR predictions against actuals.

Functions:
    score_hr_yesterday(date, conn)             -- join yesterday's HR predictions with actuals
    update_hr_calibration_summary(date, conn)  -- compute rolling HR Brier scores
    print_hr_tracking_summary(date, conn)      -- print HR tracking results
"""

import logging
import sqlite3
from datetime import datetime, timedelta

import numpy as np

from mlb_insights.config import BRIER_ROLLING_WINDOWS
from mlb_insights.signal_engine.calibration import brier_score, brier_skill_score

logger = logging.getLogger(__name__)


def score_hr_yesterday(date_str: str, conn: sqlite3.Connection) -> int:
    """Score yesterday's HR predictions against actual batter_stats results.

    Looks up hr_prediction_tracking rows for yesterday (date - 1 day), joins
    with batter_stats actuals, and updates the actual_hr and hr_correct columns.

    Args:
        date_str: Today's date (YYYY-MM-DD). Yesterday = date - 1.
        conn: Open sqlite3 connection.

    Returns:
        Number of predictions scored.
    """
    yesterday = (
        datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    # Get yesterday's HR predictions that still lack actuals
    preds = conn.execute("""
        SELECT prediction_date, player_id, p_hr
        FROM hr_prediction_tracking
        WHERE prediction_date = ? AND actual_hr IS NULL
    """, (yesterday,)).fetchall()

    if not preds:
        logger.info("No unscored HR predictions for %s.", yesterday)
        return 0

    # Get actual batter stats for yesterday
    actuals = {}
    rows = conn.execute("""
        SELECT batter_id, hr
        FROM batter_stats
        WHERE game_date = ?
    """, (yesterday,)).fetchall()
    for r in rows:
        actuals[r["batter_id"]] = r["hr"] or 0

    scored = 0
    for pred in preds:
        pid = pred["player_id"]
        if pid not in actuals:
            continue

        hr = actuals[pid]
        hr_correct = 1 if hr >= 1 else 0

        conn.execute("""
            UPDATE hr_prediction_tracking
            SET actual_hr = ?, hr_correct = ?
            WHERE prediction_date = ? AND player_id = ?
        """, (hr, hr_correct, yesterday, pid))
        scored += 1

    conn.commit()
    logger.info("Scored %d/%d HR predictions for %s.", scored, len(preds), yesterday)
    return scored


def update_hr_calibration_summary(date_str: str, conn: sqlite3.Connection):
    """Compute rolling HR Brier scores and top-N accuracy, save to calibration_summary.

    For each window in BRIER_ROLLING_WINDOWS, computes Brier score, BSS,
    and top-5/10/15 accuracy for HR predictions.

    Args:
        date_str: Current date.
        conn: Open sqlite3 connection.
    """
    for window in BRIER_ROLLING_WINDOWS:
        start_date = (
            datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=window)
        ).strftime("%Y-%m-%d")

        rows = conn.execute("""
            SELECT p_hr, actual_hr, hr_rank
            FROM hr_prediction_tracking
            WHERE prediction_date > ? AND prediction_date <= ?
              AND actual_hr IS NOT NULL
            ORDER BY prediction_date, hr_rank
        """, (start_date, date_str)).fetchall()

        if len(rows) < 10:
            continue

        phr = np.array([r["p_hr"] for r in rows])
        yhr = np.array([1.0 if (r["actual_hr"] or 0) >= 1 else 0.0 for r in rows])

        hr_brier = brier_score(phr, yhr)
        hr_bss = brier_skill_score(phr, yhr)

        # Top-N accuracy: group by prediction_date, take top N per day
        daily_groups = {}
        for r in rows:
            date_key = r["prediction_date"] if "prediction_date" in r.keys() else None
            if date_key is None:
                continue
            daily_groups.setdefault(date_key, []).append(r)

        def _top_n_accuracy(n: int) -> float:
            hits = 0
            total = 0
            for day_rows in daily_groups.values():
                top = [r for r in day_rows if r["hr_rank"] <= n]
                for r in top:
                    total += 1
                    if (r["actual_hr"] or 0) >= 1:
                        hits += 1
            return hits / total if total > 0 else 0.0

        hr_top5_accuracy = _top_n_accuracy(5)
        hr_top10_accuracy = _top_n_accuracy(10)
        hr_top15_accuracy = _top_n_accuracy(15)

        metrics = [
            ("hr_brier", hr_brier),
            ("hr_bss", hr_bss),
            ("hr_top5_accuracy", hr_top5_accuracy),
            ("hr_top10_accuracy", hr_top10_accuracy),
            ("hr_top15_accuracy", hr_top15_accuracy),
        ]

        for metric_type, value in metrics:
            conn.execute("""
                INSERT OR REPLACE INTO calibration_summary
                    (summary_date, metric_type, window_days, value, sample_size)
                VALUES (?,?,?,?,?)
            """, (date_str, metric_type, window, value, len(rows)))

    conn.commit()
    logger.info("Updated HR calibration summary for %s.", date_str)


def print_hr_tracking_summary(date_str: str, conn: sqlite3.Connection):
    """Print a summary of recent HR prediction accuracy to stdout.

    Args:
        date_str: Current date.
        conn: Open sqlite3 connection.
    """
    yesterday = (
        datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    # Yesterday's HR results
    rows = conn.execute("""
        SELECT hr_rank, p_hr, actual_hr
        FROM hr_prediction_tracking
        WHERE prediction_date = ? AND actual_hr IS NOT NULL
        ORDER BY hr_rank
    """, (yesterday,)).fetchall()

    if not rows:
        print(f"  No scored HR predictions for {yesterday}.")
        return

    total = len(rows)
    hr_rate = sum(1 for r in rows if (r["actual_hr"] or 0) >= 1) / total

    # Top 5 performance
    top5 = [r for r in rows if r["hr_rank"] <= 5]
    top5_hr = sum(1 for r in top5 if (r["actual_hr"] or 0) >= 1) / len(top5) if top5 else 0

    # Top 10 performance
    top10 = [r for r in rows if r["hr_rank"] <= 10]
    top10_hr = sum(1 for r in top10 if (r["actual_hr"] or 0) >= 1) / len(top10) if top10 else 0

    # Top 15 performance
    top15 = [r for r in rows if r["hr_rank"] <= 15]
    top15_hr = sum(1 for r in top15 if (r["actual_hr"] or 0) >= 1) / len(top15) if top15 else 0

    print(f"\n  HR Tracking Summary for {yesterday}:")
    print(f"    Total scored: {total}")
    print(f"    Overall HR rate: {hr_rate:.3f}")
    if top5:
        print(f"    Top 5 HR accuracy: {top5_hr:.3f} ({sum(1 for r in top5 if (r['actual_hr'] or 0) >= 1)}/{len(top5)})")
    if top10:
        print(f"    Top 10 HR accuracy: {top10_hr:.3f} ({sum(1 for r in top10 if (r['actual_hr'] or 0) >= 1)}/{len(top10)})")
    if top15:
        print(f"    Top 15 HR accuracy: {top15_hr:.3f} ({sum(1 for r in top15 if (r['actual_hr'] or 0) >= 1)}/{len(top15)})")

    # Rolling calibration summary (HR-specific metrics only)
    cal = conn.execute("""
        SELECT metric_type, window_days, value, sample_size
        FROM calibration_summary
        WHERE summary_date = ? AND metric_type LIKE 'hr_%'
        ORDER BY window_days, metric_type
    """, (date_str,)).fetchall()

    if cal:
        print(f"\n  HR Calibration Summary (as of {date_str}):")
        for r in cal:
            print(f"    {r['metric_type']} ({r['window_days']}d): {r['value']:.6f} (n={r['sample_size']})")
