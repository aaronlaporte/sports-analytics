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
    score_yesterday, score_hr_watch, update_calibration_summary,
    print_tracking_summary,
)
from mlb_insights.signal_engine.hr_model import (
    compute_hr_features, write_hr_features, compute_park_factors,
    write_park_factors, check_hr_power_signal,
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


# ── Per-Player Probability Model ─────────────────────────────────────────────

# League average base rates
_LEAGUE_AVG_1HIT = 0.650   # ~65% of games a qualified batter gets 1+ hit
_LEAGUE_AVG_2HIT = 0.227   # ~22.7% of games get 2+ hits
_LEAGUE_AVG_HR = 0.040     # ~4% of games a batter hits a HR

# Avg PA per game for probability conversion
_AVG_PA_PER_GAME = 3.8


def _compute_player_probabilities(
    ctx: SignalContext, batter_id: int, date_str: str
) -> tuple[float, float, float]:
    """Compute per-player P(1+ hit), P(2+ hit), P(HR) from batter_stats.

    Uses the player's season batting average and recent form to produce
    differentiated probabilities instead of a flat league average.

    The model:
    - Estimates per-AB hit probability from a blend of season avg and recent form
    - Converts to per-game probability: P(1+ hit) = 1 - (1 - p_hit)^PA
    - P(2+ hit) uses binomial: 1 - (1-p)^n - n*p*(1-p)^(n-1)

    Returns:
        (p_1hit, p_2hit, p_hr) — clamped to reasonable bounds.
    """
    from mlb_insights.signal_engine.signals import _get_batter_stats

    bs = _get_batter_stats(ctx, batter_id, date_str)
    if not bs:
        return _LEAGUE_AVG_1HIT, _LEAGUE_AVG_2HIT, _LEAGUE_AVG_HR

    season_hit_pct = bs.get("season_hit_pct", 0) or 0
    season_pa = bs.get("season_pa", 0) or 0
    hits_pg_5 = bs.get("hits_pg_5", 0) or 0
    hits_pg_10 = bs.get("hits_pg_10", 0) or 0

    if season_pa < 20 or season_hit_pct <= 0:
        return _LEAGUE_AVG_1HIT, _LEAGUE_AVG_2HIT, _LEAGUE_AVG_HR

    # Blend season avg with recent form for per-AB hit probability
    # Recent form: hits in last 5 games / approximate AB in 5 games
    recent_ab_approx = 5 * _AVG_PA_PER_GAME * 0.88  # ~16.7 AB in 5 games
    recent_hit_rate = hits_pg_5 / recent_ab_approx if recent_ab_approx > 0 else 0

    # Weight: season gets more weight as sample grows, recent form adds volatility
    season_weight = min(season_pa / 200.0, 0.70)  # caps at 70% weight
    recent_weight = 1.0 - season_weight

    # Blended per-AB hit probability
    p_hit_ab = season_weight * season_hit_pct + recent_weight * recent_hit_rate

    # Clamp to reasonable range (no one hits .400+ sustained, floor at .100)
    p_hit_ab = max(0.100, min(0.380, p_hit_ab))

    # Convert per-AB to per-game probabilities
    # Assuming ~3.8 PA per game, ~88% are AB
    ab_per_game = _AVG_PA_PER_GAME * 0.88  # ~3.34 AB

    # P(1+ hit in game) = 1 - P(0 hits) = 1 - (1 - p)^n
    p_1hit = 1.0 - (1.0 - p_hit_ab) ** ab_per_game

    # P(2+ hits in game) = 1 - P(0 hits) - P(exactly 1 hit)
    # P(exactly 1) = C(n,1) * p * (1-p)^(n-1)
    p_exactly_1 = ab_per_game * p_hit_ab * (1.0 - p_hit_ab) ** (ab_per_game - 1)
    p_2hit = 1.0 - (1.0 - p_hit_ab) ** ab_per_game - p_exactly_1

    # P(HR) — rough estimate from season HR rate, will be overridden by HR model
    season_hits = bs.get("season_hits", 0) or 0
    hr_from_stats = bs.get("hr", 0) or 0  # today's HR count (not useful)

    # Use season HR data if available in batter_stats
    # HR rate per AB * AB per game
    # We don't have season_hr in batter_stats directly, use league avg scaled
    p_hr = _LEAGUE_AVG_HR  # Will be overridden by HR model anyway

    # Clamp to reasonable bounds
    p_1hit = max(0.30, min(0.90, p_1hit))
    p_2hit = max(0.05, min(0.55, p_2hit))
    p_hr = max(0.01, min(0.20, p_hr))

    return round(p_1hit, 4), round(p_2hit, 4), round(p_hr, 4)


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
            score_hr_watch(date_str, conn)
            update_calibration_summary(date_str, conn)
            print_tracking_summary(date_str, conn)
        except Exception as exc:
            logger.error("Tracking step failed: %s", exc)

    # Step 2: Pull Statcast for the data-through date
    if not skip_ingest:
        print(f"\n[2/8] Pulling Statcast data for {date_str} ...")
        try:
            pull_statcast(date_str, conn)
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

    # Step 3b: Compute HR features
    print("\n[3b/8] Computing HR features ...")
    hr_features_map = {}
    try:
        hr_df = compute_hr_features(conn, date_str)
        if not hr_df.empty:
            write_hr_features(conn, date_str, hr_df)
            hr_features_map = {
                int(row["batter_id"]): row for _, row in hr_df.iterrows()
            }
            print(f"  HR features computed for {len(hr_df)} batters.")
        else:
            print("  No HR features computed (no active batters).")
    except Exception as exc:
        logger.error("HR feature computation failed: %s", exc)

    # Step 4: Load calibration model
    print("\n[4/8] Loading calibration model ...")
    models = None
    try:
        models = load_calibration()
    except Exception as exc:
        logger.error("Calibration load failed: %s", exc)
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

        # Compute per-player base probabilities from batter_stats
        raw_p1, raw_p2, raw_phr = _compute_player_probabilities(ctx, bid, date_str)

        try:
            cal_p1, cal_p2, cal_phr = calibrate_single(models, raw_p1, raw_p2, raw_phr)
        except Exception:
            cal_p1, cal_p2, cal_phr = raw_p1, raw_p2, raw_phr

        # Override P(HR) with HR model output if available
        hr_feat = hr_features_map.get(bid)
        if hr_feat is not None:
            cal_phr = float(hr_feat["p_hr"])

            # Check hr_power_signal
            if check_hr_power_signal(
                float(hr_feat["barrel_rate"]),
                float(hr_feat["avg_exit_velo"]),
                float(hr_feat["pitcher_hr_vuln"]),
            ):
                from mlb_insights.signal_engine.signals import SignalResult
                hr_signal = SignalResult(
                    signal_type="hr_power_signal",
                    confidence=min(float(hr_feat["barrel_rate"]) / 0.15, 1.0),
                    headline=(
                        f"Power surge: {hr_feat['barrel_rate']:.1%} barrel rate, "
                        f"{hr_feat['avg_exit_velo']:.0f} mph exit velo"
                    ),
                    reasons=[
                        f"Barrel rate: {hr_feat['barrel_rate']:.1%} (>{10}% threshold)",
                        f"Avg exit velo: {hr_feat['avg_exit_velo']:.1f} mph (>90 threshold)",
                        f"Pitcher HR vulnerability: {hr_feat['pitcher_hr_vuln']:.2f}x league avg",
                        f"Park factor: {hr_feat['park_factor']:.2f}",
                    ],
                    interpretation="Batter showing elite power metrics against a homer-prone pitcher",
                )
                signals.append(hr_signal)

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

    # Compute park factors (once)
    print("\n[2b/5] Computing park factors ...")
    try:
        park_factors_dict = compute_park_factors(conn)
        write_park_factors(conn, park_factors_dict)
        print(f"  Park factors computed for {len(park_factors_dict)} teams.")
    except Exception as exc:
        logger.error("Park factor computation failed: %s", exc)
        park_factors_dict = {}

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

        # Compute HR features for this date
        hr_features_map = {}
        try:
            hr_df = compute_hr_features(conn, date_str, park_factors=park_factors_dict)
            if not hr_df.empty:
                write_hr_features(conn, date_str, hr_df)
                hr_features_map = {
                    int(row["batter_id"]): row for _, row in hr_df.iterrows()
                }
        except Exception as exc:
            logger.debug("HR features failed for %s: %s", date_str, exc)

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

            # Compute per-player base probabilities from batter_stats
            raw_p1, raw_p2, raw_phr = _compute_player_probabilities(ctx, bid, date_str)

            try:
                cal_p1, cal_p2, cal_phr = calibrate_single(models, raw_p1, raw_p2, raw_phr)
            except Exception:
                cal_p1, cal_p2, cal_phr = raw_p1, raw_p2, raw_phr

            # Override P(HR) with HR model output if available
            hr_feat = hr_features_map.get(bid)
            if hr_feat is not None:
                cal_phr = float(hr_feat["p_hr"])

                # Check hr_power_signal
                if check_hr_power_signal(
                    float(hr_feat["barrel_rate"]),
                    float(hr_feat["avg_exit_velo"]),
                    float(hr_feat["pitcher_hr_vuln"]),
                ):
                    from mlb_insights.signal_engine.signals import SignalResult
                    hr_signal = SignalResult(
                        signal_type="hr_power_signal",
                        confidence=min(float(hr_feat["barrel_rate"]) / 0.15, 1.0),
                        headline=(
                            f"Power surge: {hr_feat['barrel_rate']:.1%} barrel rate, "
                            f"{hr_feat['avg_exit_velo']:.0f} mph exit velo"
                        ),
                        reasons=[
                            f"Barrel rate: {hr_feat['barrel_rate']:.1%} (>10% threshold)",
                            f"Avg exit velo: {hr_feat['avg_exit_velo']:.1f} mph (>90 threshold)",
                            f"Pitcher HR vulnerability: {hr_feat['pitcher_hr_vuln']:.2f}x league avg",
                            f"Park factor: {hr_feat['park_factor']:.2f}",
                        ],
                        interpretation="Batter showing elite power metrics against a homer-prone pitcher",
                    )
                    signals.append(hr_signal)

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
        default=(datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d"),
        help="Data-through date (default: yesterday). Format: YYYY-MM-DD. "
             "The pipeline pulls data through this date and generates "
             "predictions for the following day.",
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
