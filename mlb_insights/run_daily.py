"""
mlb_insights/run_daily.py -- Main entry point for the MLB Insights daily pipeline.

Usage:
    cd sports-analytics
    python -m mlb_insights.run_daily --date 2025-04-01
    python -m mlb_insights.run_daily --backfill --start 2024-03-20 --end 2025-10-24

Pipeline steps:
    1. Score yesterday's predictions against actuals
    2. Pull today's Statcast data (if available)
    3. Build/update feature tables
    4. Load calibration model (train if missing)
    5. Evaluate all signals for today's batters
    6. Compute composite scores and rankings
    7. Write daily_leaderboard, daily_signals, prediction_tracking
    8. Generate player page data
    9. Print summary to stdout
"""

import argparse
import logging
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd

from mlb_insights.config import (
    DB_PATH, LEADERBOARD_TOP_N,
    MLB_SEASON_START_2024,
)
from mlb_insights.utils.db import get_connection, ensure_tables
from mlb_insights.data_pipeline.ingest import pull_statcast, pull_schedule
from mlb_insights.data_pipeline.features import (
    build_batter_features, write_batter_features,
    build_pitcher_features, write_pitcher_features,
    build_matchup_features, write_matchup_features,
)
from mlb_insights.signal_engine.calibration import (
    load_calibration, train_calibration, calibrate_single,
)
from mlb_insights.signal_engine.signals import (
    build_signal_context, evaluate_all_signals, load_batter_date_pitchers,
    SignalContext, SignalResult,
)
from mlb_insights.signal_engine.composite import (
    compute_composite_score, rank_players, rank_players_global,
    ScoredPlayer,
)
from mlb_insights.outputs.leaderboard import (
    write_leaderboard, write_signals, write_prediction_tracking,
    print_leaderboard,
)
from mlb_insights.outputs.player_page import write_player_pages
from mlb_insights.outputs.tracking import (
    score_yesterday, update_calibration_summary, print_tracking_summary,
)

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False):
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ── Single-Date Pipeline ─────────────────────────────────────────────────────

def run_single_date(
    date_str: str,
    conn: sqlite3.Connection,
    skip_ingest: bool = False,
    skip_features: bool = False,
    skip_tracking: bool = False,
    top_n: int = LEADERBOARD_TOP_N,
):
    """Run the full insights pipeline for a single date.

    Args:
        date_str: Prediction date (YYYY-MM-DD).
        conn: Open sqlite3 connection.
        skip_ingest: Skip Statcast data pull.
        skip_features: Skip feature rebuild.
        skip_tracking: Skip scoring yesterday's predictions.
        top_n: Number of players for leaderboard.
    """
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  MLB INSIGHTS PIPELINE -- {date_str}")
    print(f"{'='*60}")

    # Step 1: Score yesterday
    if not skip_tracking:
        print("\n[1/8] Scoring yesterday's predictions ...")
        try:
            scored = score_yesterday(date_str, conn)
            update_calibration_summary(date_str, conn)
            print_tracking_summary(date_str, conn)
        except Exception as exc:
            logger.error("Tracking step failed: %s", exc)

    # Step 2: Pull Statcast
    if not skip_ingest:
        print("\n[2/8] Pulling Statcast data ...")
        try:
            yesterday = (
                datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
            ).strftime("%Y-%m-%d")
            pull_statcast(yesterday, conn)
        except Exception as exc:
            logger.error("Statcast pull failed: %s", exc)
    else:
        print("\n[2/8] Skipping Statcast ingest.")

    # Step 3: Build features
    if not skip_features:
        print("\n[3/8] Building feature tables ...")
        try:
            batter_df = build_batter_features(conn, start_date=MLB_SEASON_START_2024)
            write_batter_features(conn, batter_df)
            pitcher_df = build_pitcher_features(conn, start_date=MLB_SEASON_START_2024)
            write_pitcher_features(conn, pitcher_df)
            matchup_df = build_matchup_features(conn, start_date=MLB_SEASON_START_2024)
            write_matchup_features(conn, matchup_df)
        except Exception as exc:
            logger.error("Feature build failed: %s", exc)
    else:
        print("\n[3/8] Skipping feature build.")

    # Step 4: Load calibration model
    print("\n[4/8] Loading calibration model ...")
    models = load_calibration()
    if models is None:
        print("  No calibration model found. Training from scratch ...")
        try:
            models = train_calibration(conn)
        except Exception as exc:
            logger.error("Calibration training failed: %s", exc)
            print("  FATAL: Cannot proceed without calibration. Exiting.")
            return

    # Step 5: Evaluate signals
    print("\n[5/8] Evaluating signals ...")
    ctx = build_signal_context(conn)
    batter_date_pitchers = load_batter_date_pitchers(conn)

    # Get active batters for this date from prediction_tracking or batter_stats
    # First try: players from today's daily_picks
    active_batters = conn.execute("""
        SELECT DISTINCT batter_id FROM batter_stats WHERE game_date = ?
    """, (date_str,)).fetchall()

    if not active_batters:
        # Fall back to most recent date with batter_stats
        recent = conn.execute("""
            SELECT DISTINCT batter_id FROM batter_stats
            WHERE game_date = (SELECT MAX(game_date) FROM batter_stats WHERE game_date <= ?)
        """, (date_str,)).fetchall()
        active_batters = recent

    batter_ids = [r["batter_id"] for r in active_batters]
    logger.info("Evaluating signals for %d batters on %s.", len(batter_ids), date_str)

    # Pre-compute pitcher vulnerability for this date
    from mlb_insights.signal_engine.signals import _get_pitcher_vulnerability
    pitcher_vuln = _get_pitcher_vulnerability(ctx, date_str)

    all_scored = []
    for bid in batter_ids:
        pitchers_faced = batter_date_pitchers.get((bid, date_str), set())
        signals = evaluate_all_signals(
            ctx, bid, date_str,
            pitchers_faced=pitchers_faced if pitchers_faced else None,
            pitcher_vuln=pitcher_vuln,
        )

        # Get calibrated probabilities from prediction_tracking or compute defaults
        pt = conn.execute("""
            SELECT p_1hit, p_2hit, p_hr FROM prediction_tracking
            WHERE player_id = ? AND prediction_date = ?
        """, (bid, date_str)).fetchone()

        if pt and pt["p_1hit"] is not None:
            raw_p1 = pt["p_1hit"]
            raw_p2 = pt["p_2hit"] or 0.0
            raw_phr = pt["p_hr"] or 0.0
        else:
            # Use daily_picks if available
            dp = conn.execute("""
                SELECT model_prob FROM daily_picks
                WHERE batter_id = ? AND pick_date = ?
            """, (bid, date_str)).fetchone()
            raw_p1 = dp["model_prob"] if dp else 0.65  # default base rate
            raw_p2 = raw_p1 * 0.35  # approximate
            raw_phr = raw_p1 * 0.04  # approximate

        try:
            cal_p1, cal_p2, cal_phr = calibrate_single(models, raw_p1, raw_p2, raw_phr)
        except Exception:
            cal_p1, cal_p2, cal_phr = raw_p1, raw_p2, raw_phr

        composite_raw = compute_composite_score(cal_p1, cal_p2, cal_phr, signals)

        all_scored.append(ScoredPlayer(
            date=date_str,
            player_id=bid,
            composite_raw=composite_raw,
            daily_score=0.0,
            cal_p1hit=cal_p1,
            cal_p2hit=cal_p2,
            cal_phr=cal_phr,
            signals=signals,
        ))

    # Step 6: Rank
    print(f"\n[6/8] Ranking {len(all_scored)} players ...")
    all_scored = rank_players(all_scored)

    signal_count = sum(p.active_signal_count for p in all_scored)
    print(f"  {signal_count} signals fired across {len(all_scored)} batters.")

    # Step 7: Write outputs
    print(f"\n[7/8] Writing leaderboard, signals, and predictions ...")
    write_leaderboard(conn, date_str, all_scored, top_n=top_n)
    write_signals(conn, date_str, all_scored)
    write_prediction_tracking(conn, date_str, all_scored)

    # Step 8: Player pages
    print(f"\n[8/8] Generating player pages ...")
    try:
        write_player_pages(conn, date_str)
    except Exception as exc:
        logger.error("Player page generation failed: %s", exc)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print_leaderboard(all_scored, date_str, top_n=25)
    print(f"\n  Pipeline complete in {elapsed:.1f}s.")
    print(f"{'='*60}")


# ── Backfill Pipeline ────────────────────────────────────────────────────────

def run_backfill(
    start_date: str,
    end_date: str,
    conn: sqlite3.Connection,
    global_normalization: bool = True,
    top_n: int = LEADERBOARD_TOP_N,
):
    """Run the pipeline across a date range for historical backfill.

    Unlike single-date mode, backfill:
    - Builds features once up front (not per-date)
    - Uses global normalization across all dates
    - Skips Statcast ingest (data should already be present)

    Args:
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        conn: Open sqlite3 connection.
        global_normalization: If True, normalize scores globally.
        top_n: Leaderboard size.
    """
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  MLB INSIGHTS BACKFILL: {start_date} -> {end_date}")
    print(f"{'='*60}")

    ensure_tables()

    # Build features once
    print("\n[1/5] Building feature tables ...")
    batter_df = build_batter_features(conn, start_date=start_date)
    write_batter_features(conn, batter_df)
    pitcher_df = build_pitcher_features(conn, start_date=start_date)
    write_pitcher_features(conn, pitcher_df)
    matchup_df = build_matchup_features(conn, start_date=start_date)
    write_matchup_features(conn, matchup_df)

    # Load/train calibration
    print("\n[2/5] Loading calibration model ...")
    models = load_calibration()
    if models is None:
        print("  Training calibration model ...")
        models = train_calibration(conn)

    # Build signal context
    print("\n[3/5] Building signal context ...")
    ctx = build_signal_context(conn)
    batter_date_pitchers = load_batter_date_pitchers(conn)

    # Get all dates with batter_stats
    all_dates = conn.execute("""
        SELECT DISTINCT game_date FROM batter_stats
        WHERE game_date >= ? AND game_date <= ?
        ORDER BY game_date
    """, (start_date, end_date)).fetchall()
    dates_list = [r["game_date"] for r in all_dates]

    print(f"\n[4/5] Processing {len(dates_list)} dates ...")

    all_scored = []
    pitcher_vuln_cache = {}

    for idx, date_str in enumerate(dates_list):
        if idx % 50 == 0:
            print(f"  Processing {idx + 1}/{len(dates_list)}: {date_str}")

        # Get active batters for this date
        batters = conn.execute("""
            SELECT DISTINCT batter_id FROM batter_stats WHERE game_date = ?
        """, (date_str,)).fetchall()
        batter_ids = [r["batter_id"] for r in batters]

        # Pitcher vulnerability for this date
        if date_str not in pitcher_vuln_cache:
            from mlb_insights.signal_engine.signals import _get_pitcher_vulnerability
            pitcher_vuln_cache[date_str] = _get_pitcher_vulnerability(ctx, date_str)
        pitcher_vuln = pitcher_vuln_cache[date_str]

        # Get actuals for scoring
        actuals_map = {}
        actuals = conn.execute("""
            SELECT batter_id, hits, hr, pa FROM batter_stats WHERE game_date = ?
        """, (date_str,)).fetchall()
        for a in actuals:
            actuals_map[a["batter_id"]] = {
                "hits": a["hits"], "hr": a["hr"], "pa": a["pa"],
            }

        for bid in batter_ids:
            pitchers_faced = batter_date_pitchers.get((bid, date_str), set())
            signals = evaluate_all_signals(
                ctx, bid, date_str,
                pitchers_faced=pitchers_faced if pitchers_faced else None,
                pitcher_vuln=pitcher_vuln,
            )

            # Get probabilities from prediction_tracking
            pt = conn.execute("""
                SELECT p_1hit, p_2hit, p_hr FROM prediction_tracking
                WHERE player_id = ? AND prediction_date = ?
            """, (bid, date_str)).fetchone()

            if pt and pt["p_1hit"] is not None:
                raw_p1, raw_p2, raw_phr = pt["p_1hit"], pt["p_2hit"] or 0.0, pt["p_hr"] or 0.0
            else:
                dp = conn.execute("""
                    SELECT model_prob FROM daily_picks
                    WHERE batter_id = ? AND pick_date = ?
                """, (bid, date_str)).fetchone()
                raw_p1 = dp["model_prob"] if dp else 0.65
                raw_p2 = raw_p1 * 0.35
                raw_phr = raw_p1 * 0.04

            try:
                cal_p1, cal_p2, cal_phr = calibrate_single(models, raw_p1, raw_p2, raw_phr)
            except Exception:
                cal_p1, cal_p2, cal_phr = raw_p1, raw_p2, raw_phr

            composite_raw = compute_composite_score(cal_p1, cal_p2, cal_phr, signals)

            actual = actuals_map.get(bid)
            sp = ScoredPlayer(
                date=date_str,
                player_id=bid,
                composite_raw=composite_raw,
                daily_score=0.0,
                cal_p1hit=cal_p1,
                cal_p2hit=cal_p2,
                cal_phr=cal_phr,
                signals=signals,
                actual_hits=actual["hits"] if actual else None,
                actual_hr=actual["hr"] if actual else None,
                actual_pa=actual["pa"] if actual else None,
            )
            all_scored.append(sp)

    # Rank all
    print(f"\n[5/5] Ranking {len(all_scored)} player-dates ...")
    if global_normalization:
        all_scored = rank_players_global(all_scored)
    else:
        from mlb_insights.signal_engine.composite import rank_players_by_date
        all_scored = rank_players_by_date(all_scored)

    # Write to database
    print("  Writing to database ...")
    dates_processed = set()
    for p in all_scored:
        dates_processed.add(p.date)

    for date_str in sorted(dates_processed):
        date_players = [p for p in all_scored if p.date == date_str]
        write_leaderboard(conn, date_str, date_players, top_n=top_n)
        write_signals(conn, date_str, date_players)
        write_prediction_tracking(conn, date_str, date_players)

    elapsed = time.time() - t0
    total_signals = sum(p.active_signal_count for p in all_scored)
    print(f"\n{'='*60}")
    print(f"  Backfill complete: {len(dates_processed)} dates, {len(all_scored)} entries.")
    print(f"  {total_signals} total signals fired.")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    """CLI entry point. Parses arguments and dispatches to run_single_date or run_backfill."""
    parser = argparse.ArgumentParser(
        description="MLB Player Insights Daily Pipeline",
        prog="python -m mlb_insights.run_daily",
    )
    parser.add_argument(
        "--date",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="Prediction date (default: today). Format: YYYY-MM-DD",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Run in backfill mode across a date range.",
    )
    parser.add_argument(
        "--start",
        default=MLB_SEASON_START_2024,
        help="Backfill start date (default: 2024-03-20).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Backfill end date (default: yesterday).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=LEADERBOARD_TOP_N,
        help=f"Leaderboard size (default: {LEADERBOARD_TOP_N}).",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip Statcast data pull.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature table rebuild.",
    )
    parser.add_argument(
        "--skip-tracking",
        action="store_true",
        help="Skip scoring yesterday's predictions.",
    )
    parser.add_argument(
        "--train-calibration",
        action="store_true",
        help="Force re-training of calibration model.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    # Ensure Phase 3 tables exist
    ensure_tables()

    conn = get_connection()

    if args.train_calibration:
        print("Training calibration model ...")
        train_calibration(conn)
        if not args.backfill and args.date == datetime.today().strftime("%Y-%m-%d"):
            # Just training, no pipeline run
            conn.close()
            return

    if args.backfill:
        end = args.end or (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        run_backfill(
            start_date=args.start,
            end_date=end,
            conn=conn,
            top_n=args.top,
        )
    else:
        run_single_date(
            date_str=args.date,
            conn=conn,
            skip_ingest=args.skip_ingest,
            skip_features=args.skip_features,
            skip_tracking=args.skip_tracking,
            top_n=args.top,
        )

    conn.close()


if __name__ == "__main__":
    main()
