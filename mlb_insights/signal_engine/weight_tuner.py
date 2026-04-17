"""
mlb_insights/signal_engine/weight_tuner.py -- Weekly signal weight tuning from actuals.

Analyzes how well each signal predicts hits by joining daily_signals with
prediction_tracking.  Suggests adjusted weights based on each signal's
observed "lift" over the baseline hit rate, then writes the result to
data/tuned_weights.json for the daily pipeline to pick up.

Usage:
    cd sports-analytics
    python -m mlb_insights.signal_engine.weight_tuner
    python -m mlb_insights.signal_engine.weight_tuner --lookback 60
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path

from mlb_insights.config import SIGNAL_WEIGHTS, DATA_DIR

logger = logging.getLogger(__name__)

TUNED_WEIGHTS_PATH = DATA_DIR / "tuned_weights.json"
TUNED_THRESHOLDS_PATH = DATA_DIR / "tuned_thresholds.json"

# Damping factor: controls how aggressively weights shift toward observed lift.
# 0.3 means new_weight = current * (1 + 0.3 * lift).
_DAMPING = 0.3

# Hard clamps so no single signal dominates or vanishes.
_WEIGHT_FLOOR = 0.02
_WEIGHT_CEIL = 0.15


def compute_signal_accuracy(
    conn: sqlite3.Connection,
    lookback_days: int = 90,
) -> dict[str, dict]:
    """Compute per-signal accuracy stats over the lookback window.

    For each signal type in daily_signals, joins with prediction_tracking
    on (signal_date = prediction_date, player_id) to determine whether the
    player got 1+ hit on the day the signal fired.

    Returns a dict keyed by signal_type with:
        fire_count, hit_rate_when_fired, avg_confidence, baseline_rate
    """
    # Baseline hit rate: fraction of all prediction_tracking rows where
    # the player recorded 1+ hit, regardless of signals.
    baseline_row = conn.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN actual_hits >= 1 THEN 1 ELSE 0 END) AS hits
        FROM prediction_tracking
        WHERE prediction_date >= date('now', ?)
    """, (f"-{lookback_days} days",)).fetchone()

    total = baseline_row["total"] or 0
    baseline_rate = (baseline_row["hits"] / total) if total > 0 else 0.0

    # Per-signal stats
    rows = conn.execute("""
        SELECT
            ds.signal_type,
            COUNT(*) AS fire_count,
            SUM(CASE WHEN pt.actual_hits >= 1 THEN 1 ELSE 0 END) AS hit_count,
            AVG(ds.confidence) AS avg_confidence
        FROM daily_signals ds
        JOIN prediction_tracking pt
            ON ds.signal_date = pt.prediction_date
            AND ds.player_id = pt.player_id
        WHERE ds.signal_date >= date('now', ?)
        GROUP BY ds.signal_type
    """, (f"-{lookback_days} days",)).fetchall()

    stats: dict[str, dict] = {}
    for r in rows:
        fire_count = r["fire_count"]
        hit_count = r["hit_count"] or 0
        hit_rate = hit_count / fire_count if fire_count > 0 else 0.0
        stats[r["signal_type"]] = {
            "fire_count": fire_count,
            "hit_rate_when_fired": round(hit_rate, 4),
            "avg_confidence": round(r["avg_confidence"], 4),
            "baseline_rate": round(baseline_rate, 4),
        }

    # Ensure every configured signal appears even if it never fired.
    for sig in SIGNAL_WEIGHTS:
        if sig not in stats:
            stats[sig] = {
                "fire_count": 0,
                "hit_rate_when_fired": 0.0,
                "avg_confidence": 0.0,
                "baseline_rate": round(baseline_rate, 4),
            }

    return stats


def suggest_weights(
    signal_stats: dict[str, dict],
    current_weights: dict[str, float],
) -> dict[str, float]:
    """Suggest adjusted weights based on each signal's lift over baseline.

    Lift = (hit_rate_when_fired - baseline_rate) / baseline_rate

    new_weight = current_weight * (1 + DAMPING * lift)

    Weights are clamped to [WEIGHT_FLOOR, WEIGHT_CEIL], then normalized so
    the total weight budget matches the original sum.
    """
    original_budget = sum(current_weights.values())
    suggested: dict[str, float] = {}

    for sig, cur_w in current_weights.items():
        st = signal_stats.get(sig)
        if st is None or st["fire_count"] == 0 or st["baseline_rate"] <= 0:
            # No data -- keep current weight unchanged.
            suggested[sig] = cur_w
            continue

        lift = (st["hit_rate_when_fired"] - st["baseline_rate"]) / st["baseline_rate"]
        new_w = cur_w * (1.0 + _DAMPING * lift)
        new_w = max(_WEIGHT_FLOOR, min(_WEIGHT_CEIL, new_w))
        suggested[sig] = new_w

    # Normalize to preserve original budget.
    raw_total = sum(suggested.values())
    if raw_total > 0:
        scale = original_budget / raw_total
        suggested = {k: round(v * scale, 6) for k, v in suggested.items()}

    return suggested


def print_weight_report(
    signal_stats: dict[str, dict],
    current_weights: dict[str, float],
    suggested_weights: dict[str, float],
) -> None:
    """Print a human-readable comparison table."""
    header = (
        f"{'signal_type':<30} {'fires':>6} {'hit_rate':>9} {'baseline':>9} "
        f"{'lift':>8} {'cur_wt':>8} {'new_wt':>8}"
    )
    print("\n" + "=" * len(header))
    print("  SIGNAL WEIGHT TUNING REPORT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for sig in sorted(current_weights.keys()):
        st = signal_stats.get(sig, {})
        fires = st.get("fire_count", 0)
        hit_rate = st.get("hit_rate_when_fired", 0.0)
        baseline = st.get("baseline_rate", 0.0)
        if fires > 0 and baseline > 0:
            lift = (hit_rate - baseline) / baseline
        else:
            lift = 0.0
        cur_w = current_weights.get(sig, 0.0)
        new_w = suggested_weights.get(sig, cur_w)
        print(
            f"{sig:<30} {fires:>6} {hit_rate:>8.3f}  {baseline:>8.3f} "
            f"{lift:>+7.2f}  {cur_w:>7.4f}  {new_w:>7.4f}"
        )

    print("-" * len(header))
    print(
        f"{'TOTAL':<30} {'':>6} {'':>9} {'':>9} {'':>8} "
        f"{sum(current_weights.values()):>7.4f}  "
        f"{sum(suggested_weights.values()):>7.4f}"
    )
    print("=" * len(header))


def compute_optimal_babip_threshold(
    conn: sqlite3.Connection,
    lookback_days: int = 180,
) -> dict:
    """Find the BABIP gap threshold that maximizes lift over baseline hit rate.

    For each candidate threshold, simulates how often the babip_regression
    signal would have fired (by comparing season BABIP vs expected BABIP in
    batter_stats), then checks whether those players got 1+ hit the next game
    via prediction_tracking.

    Args:
        conn: Open sqlite3 connection (read-only is fine).
        lookback_days: How far back to look for historical data.

    Returns:
        Dict with keys: optimal_threshold, report (list of per-threshold dicts).
    """
    from mlb_insights.config import (
        BABIP_MIN_SEASON_PA, BABIP_APPROX_HR_RATE,
        BABIP_APPROX_SO_RATE, BABIP_APPROX_AB_RATIO,
    )

    # Baseline hit rate over lookback window
    baseline_row = conn.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN actual_hits >= 1 THEN 1 ELSE 0 END) AS hits
        FROM prediction_tracking
        WHERE prediction_date >= date('now', ?)
    """, (f"-{lookback_days} days",)).fetchone()

    total = baseline_row["total"] or 0
    baseline_rate = (baseline_row["hits"] / total) if total > 0 else 0.0

    if total == 0:
        logger.warning("No prediction_tracking data in lookback window.")
        return {"optimal_threshold": 0.080, "report": []}

    # Pull batter_stats rows with enough season PA, joined with next-day actuals.
    # We compute BABIP gap inline: babip - expected_babip
    # expected_babip ~= (season_hit_pct - HR_RATE) / (AB_RATIO - SO_RATE - HR_RATE)
    # But we simplify: use (season_avg - approx_babip) as the gap proxy,
    # matching the signal logic which checks season_babip vs expected_babip.
    #
    # The signal stores babip and expected_babip in batter_stats.  We query
    # those directly if available, otherwise approximate.
    rows = conn.execute("""
        SELECT
            bs.batter_id,
            bs.game_date,
            bs.season_babip,
            bs.season_hit_pct,
            bs.season_pa,
            pt.actual_hits
        FROM batter_stats bs
        JOIN prediction_tracking pt
            ON bs.game_date = pt.prediction_date
            AND bs.batter_id = pt.player_id
        WHERE bs.game_date >= date('now', ?)
            AND bs.season_pa >= ?
            AND bs.season_babip IS NOT NULL
            AND bs.season_hit_pct IS NOT NULL
    """, (f"-{lookback_days} days", BABIP_MIN_SEASON_PA)).fetchall()

    if not rows:
        logger.warning("No batter_stats rows with BABIP data in lookback window.")
        return {"optimal_threshold": 0.080, "report": []}

    # Compute expected BABIP and gap for each row
    entries = []
    for r in rows:
        babip = r["season_babip"]
        hit_pct = r["season_hit_pct"]
        # Expected BABIP approximation (same formula as the signal)
        expected = (hit_pct - BABIP_APPROX_HR_RATE) / (
            BABIP_APPROX_AB_RATIO - BABIP_APPROX_SO_RATE - BABIP_APPROX_HR_RATE
        )
        gap = expected - babip  # positive = unlucky (signal fires when gap > threshold)
        entries.append({
            "gap": gap,
            "got_hit": 1 if (r["actual_hits"] or 0) >= 1 else 0,
        })

    thresholds = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12]
    min_fires = 20
    report = []
    best_threshold = 0.080
    best_lift = -999.0

    header = (
        f"{'threshold':>10} {'fires':>7} {'hit_rate':>9} {'baseline':>9} "
        f"{'lift':>8}"
    )
    print("\n" + "=" * len(header))
    print("  BABIP THRESHOLD TUNING REPORT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for thresh in thresholds:
        fired = [e for e in entries if e["gap"] >= thresh]
        fire_count = len(fired)
        if fire_count < min_fires:
            hit_rate = 0.0
            lift = 0.0
        else:
            hit_rate = sum(e["got_hit"] for e in fired) / fire_count
            lift = hit_rate - baseline_rate

        row_data = {
            "threshold": thresh,
            "fire_count": fire_count,
            "hit_rate": round(hit_rate, 4),
            "baseline_rate": round(baseline_rate, 4),
            "lift": round(lift, 4),
        }
        report.append(row_data)

        marker = ""
        if fire_count >= min_fires and lift > best_lift:
            best_lift = lift
            best_threshold = thresh

        print(
            f"{thresh:>10.3f} {fire_count:>7} {hit_rate:>8.3f}  "
            f"{baseline_rate:>8.3f} {lift:>+7.3f}"
        )

    # Mark best in report
    for r in report:
        r["selected"] = r["threshold"] == best_threshold

    print("-" * len(header))
    print(f"  Selected threshold: {best_threshold:.3f} (lift: {best_lift:+.3f})")
    print("=" * len(header))

    return {"optimal_threshold": best_threshold, "report": report}


def save_tuned_thresholds(thresholds: dict, path: Path | None = None) -> Path:
    """Persist tuned thresholds to JSON.

    Args:
        thresholds: Dict of threshold name -> value (e.g. {"babip_min_gap": 0.06}).
        path: Override output path (default: data/tuned_thresholds.json).

    Returns:
        Path the file was written to.
    """
    out = path or TUNED_THRESHOLDS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nTuned thresholds saved to {out}")
    return out


def save_tuned_weights(weights: dict[str, float], path: Path | None = None) -> Path:
    """Persist suggested weights to JSON.

    Args:
        weights: signal_type -> weight mapping.
        path: Override output path (default: data/tuned_weights.json).

    Returns:
        Path the file was written to.
    """
    out = path or TUNED_WEIGHTS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"\nTuned weights saved to {out}")
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tune MLB signal weights from historical accuracy.",
        prog="python -m mlb_insights.signal_engine.weight_tuner",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=90,
        help="Number of days to look back (default: 90).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print report but do not save tuned_weights.json.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from mlb_insights.utils.db import get_connection

    conn = get_connection(readonly=True)
    try:
        stats = compute_signal_accuracy(conn, lookback_days=args.lookback)
        suggested = suggest_weights(stats, SIGNAL_WEIGHTS)
        print_weight_report(stats, SIGNAL_WEIGHTS, suggested)

        if not args.dry_run:
            save_tuned_weights(suggested)
        else:
            print("\n(dry run -- weights not saved)")

        # BABIP threshold tuning
        babip_result = compute_optimal_babip_threshold(conn, lookback_days=args.lookback)
        if babip_result["report"]:
            if not args.dry_run:
                save_tuned_thresholds(
                    {"babip_min_gap": babip_result["optimal_threshold"]}
                )
            else:
                print("\n(dry run -- thresholds not saved)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
