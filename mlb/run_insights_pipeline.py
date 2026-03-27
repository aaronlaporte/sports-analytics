"""
mlb/run_insights_pipeline.py — Daily insights pipeline orchestrator.

Runs AFTER the existing run_pipeline.py (which produces daily_picks).
Reads daily_picks + batter_stats + pitcher_stats + matchup_stats,
then generates signals, multi-outcome probabilities, composite scores,
and the daily leaderboard.

Usage:
    python mlb/run_insights_pipeline.py
    python mlb/run_insights_pipeline.py --date 2026-03-27
    python mlb/run_insights_pipeline.py --top 50
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.db import get_conn, create_schema
from mlb.signals.engine import SignalEngine, signals_to_db_rows
from mlb.model.multi_outcome import compute_multi_outcomes
from mlb.leaderboard.composite_score import score_all_batters
from mlb.leaderboard.rank import (
    rank_leaderboard,
    write_leaderboard,
    write_prediction_tracking,
    print_leaderboard,
)


def load_daily_picks(conn, pick_date: str) -> pd.DataFrame:
    """Load today's daily_picks (P(1+ hit) from existing model)."""
    df = pd.read_sql(
        "SELECT * FROM daily_picks WHERE pick_date = ?",
        conn, params=(pick_date,)
    )
    if not df.empty:
        df = df.rename(columns={"model_prob": "p_1hit"})
    return df


def load_batter_features(conn, pick_date: str) -> pd.DataFrame:
    """Load the most recent batter_stats row per batter before pick_date."""
    return pd.read_sql("""
        SELECT b.*
        FROM batter_stats b
        INNER JOIN (
            SELECT batter_id, MAX(game_date) as max_date
            FROM batter_stats
            WHERE game_date < ?
            GROUP BY batter_id
        ) latest ON b.batter_id = latest.batter_id AND b.game_date = latest.max_date
    """, conn, params=(pick_date,))


def load_pitcher_stats(conn) -> pd.DataFrame:
    """Load aggregated pitcher stats (most recent per pitcher)."""
    return pd.read_sql("""
        SELECT p.*
        FROM pitcher_stats p
        INNER JOIN (
            SELECT pitcher_id, MAX(game_date) as max_date
            FROM pitcher_stats
            GROUP BY pitcher_id
        ) latest ON p.pitcher_id = latest.pitcher_id AND p.game_date = latest.max_date
    """, conn)


def load_matchup_stats(conn) -> pd.DataFrame:
    """Load all matchup stats."""
    return pd.read_sql("SELECT * FROM matchup_stats", conn)


def build_insights_frame(
    picks_df: pd.DataFrame,
    batter_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily_picks with batter features to create the insights input frame.

    If daily_picks is empty, fall back to batter_stats alone with a default P(1+ hit).
    """
    if picks_df.empty:
        # No picks yet — use batter features directly
        frame = batter_df.copy()
        frame["p_1hit"] = 0.0
        frame["expected_pa"] = frame.get("pa", 3.5)
        return frame

    # Merge picks with full batter features
    frame = picks_df.merge(
        batter_df.drop(columns=["game_date"], errors="ignore"),
        on="batter_id",
        how="left",
        suffixes=("", "_feat"),
    )
    return frame


def run_insights(pick_date: str, top_n: int = 50):
    """Execute the full insights pipeline for a given date."""
    create_schema("mlb")
    conn = get_conn("mlb")

    print(f"\n[insights] Running MLB Insights Pipeline for {pick_date}")
    print("=" * 60)

    # Step 1: Load data
    print("[insights] Loading data ...")
    picks_df = load_daily_picks(conn, pick_date)
    batter_df = load_batter_features(conn, pick_date)
    pitcher_df = load_pitcher_stats(conn)
    matchup_df = load_matchup_stats(conn)

    if batter_df.empty:
        print("[insights] No batter stats found. Run the base pipeline first.")
        conn.close()
        return

    print(f"  daily_picks: {len(picks_df)} rows")
    print(f"  batter_stats: {len(batter_df)} batters")
    print(f"  pitcher_stats: {len(pitcher_df)} pitchers")
    print(f"  matchup_stats: {len(matchup_df)} pairs")

    # Step 2: Build insights input frame
    frame = build_insights_frame(picks_df, batter_df)
    print(f"[insights] Insights frame: {len(frame)} batters")

    # Step 3: Compute multi-outcome probabilities
    print("[insights] Computing P(2+ hits) and P(HR) ...")
    multi_df = compute_multi_outcomes(frame, pitcher_df)
    frame = frame.merge(multi_df, on="batter_id", how="left")

    # Step 4: Evaluate signals
    print("[insights] Evaluating signals ...")
    engine = SignalEngine()
    all_signals = engine.evaluate_all_batters(frame, pitcher_df, matchup_df)
    signal_count = sum(len(v) for v in all_signals.values())
    print(f"  {signal_count} signals fired across {len(all_signals)} batters")

    # Write signals to DB
    signal_rows = signals_to_db_rows(pick_date, all_signals)
    conn.execute("DELETE FROM daily_signals WHERE signal_date = ?", (pick_date,))
    for row in signal_rows:
        conn.execute("""
            INSERT OR REPLACE INTO daily_signals
                (signal_date, player_id, signal_type, confidence,
                 headline, reasons_json, interpretation)
            VALUES (?,?,?,?,?,?,?)
        """, row)
    conn.commit()
    print(f"[insights] Wrote {len(signal_rows)} signal rows")

    # Step 5: Compute composite scores
    print("[insights] Computing composite scores ...")
    scored_df = score_all_batters(frame, all_signals)

    # Step 6: Rank and write leaderboard
    ranked_df = rank_leaderboard(scored_df, top_n=top_n)
    write_leaderboard(conn, pick_date, ranked_df)
    write_prediction_tracking(conn, pick_date, ranked_df)

    # Step 7: Print leaderboard
    print_leaderboard(ranked_df, pick_date, top_n=25)

    conn.close()
    print(f"\n[insights] Done. Leaderboard and signals written to mlb.db.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB Player Insights Pipeline")
    parser.add_argument("--date", default=date.today().strftime("%Y-%m-%d"),
                        help="Prediction date (default: today)")
    parser.add_argument("--top", type=int, default=50,
                        help="Number of players on leaderboard (default: 50)")
    args = parser.parse_args()
    run_insights(args.date, args.top)
