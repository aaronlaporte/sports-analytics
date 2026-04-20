"""
mlb_insights/run_hr_pipeline.py -- Dedicated HR prediction pipeline runner.

Usage:
    cd sports-analytics
    python -m mlb_insights.run_hr_pipeline --date 2025-04-01
    python -m mlb_insights.run_hr_pipeline --date 2025-04-01 --retune --verbose

Pipeline steps:
    1. Score yesterday's HR predictions against actuals
    2. Compute HR features v2
    3. Load/train HR calibration model
    4. Evaluate HR signals for all batters
    5. Build HRScoredPlayer list with composite scores
    6. Rank and filter to players with games
    7. Write HR leaderboard and prediction tracking
"""

import argparse
import logging
import sqlite3
import time
from datetime import datetime, timedelta

from mlb_insights.utils.db import get_connection, ensure_tables

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False):
    """Configure logging for the HR pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_hr_pipeline(date_str: str, conn: sqlite3.Connection):
    """Run the dedicated HR prediction pipeline for a single date."""

    t0 = time.time()
    print(f"\n  HR PIPELINE -- {date_str}")
    print(f"  {'─'*40}")

    # Step 1: Score yesterday's HR predictions
    from mlb_insights.outputs.hr_tracking import (
        score_hr_yesterday, update_hr_calibration_summary, print_hr_tracking_summary
    )
    try:
        scored = score_hr_yesterday(date_str, conn)
        update_hr_calibration_summary(date_str, conn)
        print_hr_tracking_summary(date_str, conn)
    except Exception as exc:
        logger.error("HR tracking failed: %s", exc)

    # Step 2: Compute HR features v2
    from mlb_insights.signal_engine.hr_features_v2 import (
        compute_hr_features_v2, write_hr_features_v2
    )
    try:
        hr_v2_df = compute_hr_features_v2(conn, date_str)
        if not hr_v2_df.empty:
            write_hr_features_v2(conn, date_str, hr_v2_df)
            hr_v2_map = {int(row["batter_id"]): row.to_dict() for _, row in hr_v2_df.iterrows()}
        else:
            hr_v2_map = {}
    except Exception as exc:
        logger.error("HR features v2 failed: %s", exc)
        hr_v2_map = {}

    if not hr_v2_map:
        print("  No HR features computed. Skipping HR pipeline.")
        return

    # Step 3: Load/train HR calibration
    from mlb_insights.signal_engine.hr_calibration import (
        load_hr_calibration, train_hr_calibration,
        calibrate_hr_single, hr_calibration_needs_retrain
    )
    hr_cal = load_hr_calibration()
    if hr_cal is None or hr_calibration_needs_retrain(conn):
        try:
            hr_cal = train_hr_calibration(conn)
        except Exception as exc:
            logger.error("HR calibration training failed: %s", exc)

    # Step 4: Evaluate HR signals for all batters
    from mlb_insights.signal_engine.hr_signals import (
        evaluate_hr_signals, compute_pitcher_fb_rates
    )
    from mlb_insights.signal_engine.hr_model import check_hr_power_signal

    pitcher_fb_rates = compute_pitcher_fb_rates(conn, date_str)

    # Get opposing pitcher info per batter (same approach as run_daily.py)
    # Load from statcast_staging
    batter_ids = list(hr_v2_map.keys())
    placeholders = ",".join("?" * len(batter_ids))

    # Get opposing pitchers
    opp_pitchers = conn.execute(f"""
        SELECT DISTINCT batter, pitcher, p_throws
        FROM statcast_staging
        WHERE batter IN ({placeholders}) AND game_date = ?
        AND events IS NOT NULL
    """, batter_ids + [date_str]).fetchall()

    batter_opp_pitcher = {}  # batter_id -> set of pitcher_ids
    batter_opp_hand = {}     # batter_id -> 'L' or 'R'
    for r in opp_pitchers:
        batter_opp_pitcher.setdefault(r["batter"], set()).add(r["pitcher"])
        if r["p_throws"]:
            batter_opp_hand[r["batter"]] = r["p_throws"]

    # If no statcast data for prediction date, use most recent
    if not batter_opp_pitcher:
        recent_date = conn.execute("""
            SELECT MAX(game_date) FROM statcast_staging WHERE game_date < ?
        """, (date_str,)).fetchone()
        if recent_date and recent_date[0]:
            opp_pitchers = conn.execute(f"""
                SELECT DISTINCT batter, pitcher, p_throws
                FROM statcast_staging
                WHERE batter IN ({placeholders}) AND game_date = ?
                AND events IS NOT NULL
            """, batter_ids + [recent_date[0]]).fetchall()
            for r in opp_pitchers:
                batter_opp_pitcher.setdefault(r["batter"], set()).add(r["pitcher"])
                if r["p_throws"]:
                    batter_opp_hand[r["batter"]] = r["p_throws"]

    # Step 5: Build HRScoredPlayer list
    from mlb_insights.signal_engine.hr_composite import (
        HRScoredPlayer, compute_hr_composite, rank_hr_players
    )
    from mlb_insights.signal_engine.signals import SignalResult

    all_hr_scored = []
    for bid, features in hr_v2_map.items():
        p_hr = features.get("p_hr_v2", features.get("p_hr", 0.04))

        # Apply HR calibration if available
        if hr_cal:
            try:
                p_hr = calibrate_hr_single(hr_cal, p_hr)
            except Exception:
                pass

        # Evaluate HR signals
        hr_sigs = evaluate_hr_signals(
            conn=conn,
            batter_id=bid,
            game_date=date_str,
            hr_features=features,
            pitcher_fb_rates=pitcher_fb_rates,
            opp_pitcher_ids=batter_opp_pitcher.get(bid, set()),
            opp_pitcher_hand=batter_opp_hand.get(bid),
        )

        # Also check existing hr_power_signal
        if check_hr_power_signal(
            features.get("barrel_rate", 0),
            features.get("avg_exit_velo", 0),
            features.get("pitcher_hr_vuln", 1.0),
        ):
            hr_sigs.append(SignalResult(
                signal_type="hr_power_signal",
                confidence=min(features.get("barrel_rate", 0) / 0.15, 1.0),
                headline=f"Power surge: {features.get('barrel_rate', 0):.1%} barrel rate, {features.get('avg_exit_velo', 0):.0f} mph exit velo",
                reasons=[f"Barrel rate: {features.get('barrel_rate', 0):.1%}", f"Exit velo: {features.get('avg_exit_velo', 0):.1f} mph"],
                interpretation="Elite power metrics against HR-prone pitcher",
            ))

        composite_raw = compute_hr_composite(p_hr, hr_sigs)

        all_hr_scored.append(HRScoredPlayer(
            date=date_str,
            player_id=bid,
            composite_raw=composite_raw,
            hr_score=0.0,
            p_hr=p_hr,
            signals=hr_sigs,
            barrel_rate=features.get("barrel_rate", 0),
            fly_ball_rate=features.get("fly_ball_rate", 0),
            hard_hit_fly_rate=features.get("hard_hit_fly_rate", 0),
            park_factor=features.get("park_factor", 1.0),
        ))

    # Step 6: Rank
    all_hr_scored = rank_hr_players(all_hr_scored)
    signal_count = sum(p.active_hr_signal_count for p in all_hr_scored)
    print(f"  Ranked {len(all_hr_scored)} batters, {signal_count} HR signals fired.")

    # Step 6b: Filter to players with games on prediction target date
    from mlb_insights.data_pipeline.ingest import pull_schedule

    prediction_target = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    games = pull_schedule(prediction_target, conn)
    if games:
        teams_playing = set()
        for g in games:
            if g.get("home_team"): teams_playing.add(g["home_team"])
            if g.get("away_team"): teams_playing.add(g["away_team"])

        team_map = {}
        for row in conn.execute("SELECT batter_id, team FROM batter_stats WHERE game_date = ? AND team IS NOT NULL", (date_str,)).fetchall():
            team_map[row["batter_id"]] = row["team"]
        if not team_map:
            for row in conn.execute("SELECT batter_id, team FROM batter_stats WHERE game_date = (SELECT MAX(game_date) FROM batter_stats WHERE game_date <= ?) AND team IS NOT NULL", (date_str,)).fetchall():
                team_map.setdefault(row["batter_id"], row["team"])

        pre_filter = len(all_hr_scored)
        all_hr_scored = [p for p in all_hr_scored if team_map.get(p.player_id) in teams_playing]
        all_hr_scored = rank_hr_players(all_hr_scored)
        print(f"  Filtered: {pre_filter} -> {len(all_hr_scored)} with games on {prediction_target}.")

    # Step 7: Write outputs
    from mlb_insights.outputs.hr_leaderboard import (
        write_hr_leaderboard, write_hr_prediction_tracking, print_hr_leaderboard
    )
    write_hr_leaderboard(conn, date_str, all_hr_scored, top_n=15)
    write_hr_prediction_tracking(conn, date_str, all_hr_scored)
    print_hr_leaderboard(all_hr_scored, date_str, top_n=15)

    elapsed = time.time() - t0
    print(f"  HR pipeline complete in {elapsed:.1f}s.")


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    """CLI entry point for the standalone HR pipeline."""
    parser = argparse.ArgumentParser(
        description="MLB Insights HR Prediction Pipeline",
        prog="python -m mlb_insights.run_hr_pipeline",
    )
    parser.add_argument(
        "--date",
        default=(datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d"),
        help="Data-through date (default: yesterday). Format: YYYY-MM-DD.",
    )
    parser.add_argument(
        "--retune",
        action="store_true",
        help="Force refit of HR v2 model and HR calibration before running.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    ensure_tables()
    conn = get_connection()

    if args.retune:
        print("Force-retraining HR calibration model ...")
        from mlb_insights.signal_engine.hr_calibration import train_hr_calibration
        try:
            train_hr_calibration(conn)
            print("  HR calibration retrained.")
        except Exception as exc:
            logger.error("HR calibration retrain failed: %s", exc)

        print("Force-retraining HR v2 model ...")
        from mlb_insights.signal_engine.hr_features_v2 import retrain_hr_v2_model
        try:
            retrain_hr_v2_model(conn)
            print("  HR v2 model retrained.")
        except Exception as exc:
            logger.error("HR v2 model retrain failed: %s", exc)

    run_hr_pipeline(args.date, conn)
    conn.close()


if __name__ == "__main__":
    main()
