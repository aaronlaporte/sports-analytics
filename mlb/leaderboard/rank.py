"""
mlb/leaderboard/rank.py — Rank batters by daily_score and write leaderboard to DB.
"""

from __future__ import annotations

import pandas as pd


def rank_leaderboard(scored_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Sort by daily_score descending and assign ranks.

    Args:
        scored_df: DataFrame with daily_score column (output of composite_score.score_all_batters).
        top_n:     Number of players to include in the leaderboard.

    Returns:
        DataFrame with daily_rank column, limited to top_n rows.
    """
    ranked = scored_df.sort_values("daily_score", ascending=False).head(top_n).copy()
    ranked["daily_rank"] = range(1, len(ranked) + 1)
    return ranked


def write_leaderboard(conn, prediction_date: str, ranked_df: pd.DataFrame):
    """Write the ranked leaderboard to the daily_leaderboard table."""
    conn.execute("DELETE FROM daily_leaderboard WHERE prediction_date = ?", (prediction_date,))

    for _, r in ranked_df.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO daily_leaderboard
                (prediction_date, player_id, player_name, team, opponent,
                 opp_pitcher, daily_rank, daily_score,
                 p_1hit, p_2hit, p_hr,
                 active_signal_count, top_signal, top_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            prediction_date,
            int(r.get("batter_id", 0)),
            r.get("batter_name"),
            r.get("team"),
            r.get("opponent"),
            r.get("opp_pitcher"),
            int(r.get("daily_rank", 0)),
            float(r.get("daily_score", 0)),
            float(r.get("p_1hit", r.get("model_prob", 0)) or 0),
            float(r.get("p_2hit", 0) or 0),
            float(r.get("p_hr", 0) or 0),
            int(r.get("active_signal_count", 0) or 0),
            r.get("top_signal", ""),
            r.get("top_reason", ""),
        ))

    conn.commit()
    print(f"[leaderboard] Wrote {len(ranked_df)} rows for {prediction_date}")


def write_prediction_tracking(conn, prediction_date: str, ranked_df: pd.DataFrame):
    """Write predictions to the tracking table (actuals filled in later)."""
    conn.execute("DELETE FROM prediction_tracking WHERE prediction_date = ?", (prediction_date,))

    for _, r in ranked_df.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO prediction_tracking
                (prediction_date, player_id, player_name,
                 daily_rank, daily_score, p_1hit, p_2hit, p_hr)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            prediction_date,
            int(r.get("batter_id", 0)),
            r.get("batter_name"),
            int(r.get("daily_rank", 0)),
            float(r.get("daily_score", 0)),
            float(r.get("p_1hit", r.get("model_prob", 0)) or 0),
            float(r.get("p_2hit", 0) or 0),
            float(r.get("p_hr", 0) or 0),
        ))

    conn.commit()
    print(f"[tracking] Wrote {len(ranked_df)} prediction rows for {prediction_date}")


def print_leaderboard(ranked_df: pd.DataFrame, prediction_date: str, top_n: int = 25):
    """Print the leaderboard to stdout."""
    print(f"\n{'='*80}")
    print(f"  MLB PLAYER INSIGHTS — LEADERBOARD for {prediction_date}")
    print(f"{'='*80}")
    print(f" {'#':>3}  {'Player':<24} {'Team':<5} {'Score':>6}  {'P(1H)':>6}  "
          f"{'P(2H)':>6}  {'P(HR)':>6}  {'Signals'}")
    print("-" * 80)

    for _, r in ranked_df.head(top_n).iterrows():
        rank = int(r.get("daily_rank", 0))
        name = str(r.get("batter_name", ""))[:23]
        team = str(r.get("team", ""))[:4]
        score = float(r.get("daily_score", 0))
        p1 = float(r.get("p_1hit", r.get("model_prob", 0)) or 0)
        p2 = float(r.get("p_2hit", 0) or 0)
        p_hr = float(r.get("p_hr", 0) or 0)
        sig_count = int(r.get("active_signal_count", 0) or 0)
        top_sig = str(r.get("top_signal", ""))

        sig_str = top_sig if sig_count > 0 else "--"
        if sig_count > 1:
            sig_str += f" (+{sig_count - 1} more)"

        print(f" {rank:>3}  {name:<24} {team:<5} {score:>6.3f}  {p1:>6.3f}  "
              f"{p2:>6.3f}  {p_hr:>6.3f}  {sig_str}")
