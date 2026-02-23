"""
mlb/predict/daily_picks.py — Generate today's top hit probability picks with auto EV.

Loads model params from mlb.db, scores eligible batters, joins with live odds
from The Odds API, computes EV, and writes results to daily_picks table.

Usage:
    python mlb/predict/daily_picks.py
    python mlb/predict/daily_picks.py --date 2025-04-15 --top 10
"""

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.db import get_conn, create_schema

# American odds → decimal
def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1


def ev(prob: float, decimal_odds: float) -> float:
    return round(prob * decimal_odds - 1, 4)


def load_model_params(conn) -> dict:
    row = conn.execute(
        "SELECT params_json FROM model_params ORDER BY run_date DESC LIMIT 1"
    ).fetchone()
    if not row:
        raise RuntimeError("No model params found. Run mlb/model/train_model.py first.")
    return json.loads(row["params_json"])


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


def load_live_odds(conn) -> pd.DataFrame:
    """Load most recent batter_hits Over lines per player."""
    return pd.read_sql("""
        SELECT player, line, price, bookmaker,
               MAX(fetched_at) as fetched_at
        FROM live_odds
        WHERE market = 'batter_hits'
          AND name   = 'Over'
        GROUP BY player, line
        ORDER BY price DESC
    """, conn)


def compute_hit_prob(row: pd.Series, params: dict) -> float:
    """Beta-binomial shrinkage model (ported from mlb-betting-analytics)."""
    league_rate    = params["league_hit_per_pa"]
    season_prior   = params["season_prior_pa"]
    recent_weight  = params["recent_weight"]
    split_weight   = params.get("split_weight", 0.0)

    season_hits = row.get("season_hits", 0) or 0
    season_pa   = row.get("season_pa",   0) or 0
    recent_hits = row.get("hits_pg_5",   0) or 0

    # Season shrinkage rate
    season_rate = (season_hits + league_rate * season_prior) / (season_pa + season_prior)

    # Recent form blending
    recent_pa = 5 * max(row.get("pa", 3), 1)  # approx PA in last 5 games
    if recent_pa > 0:
        recent_rate = (recent_hits + league_rate * 10) / (recent_pa + 10)
    else:
        recent_rate = season_rate

    base_weight = 1.0 - recent_weight - split_weight
    blended = base_weight * season_rate + recent_weight * recent_rate

    # Expected PA this game
    expected_pa = min(max(row.get("pa", 3.5), 3.0), 5.0)

    # P(at least 1 hit)
    p_miss = (1 - blended) ** expected_pa
    return round(1 - p_miss, 4)


def run_picks(pick_date: str, top_n: int = 10):
    create_schema("mlb")
    conn = get_conn("mlb")

    try:
        params = load_model_params(conn)
    except RuntimeError as e:
        print(f"[picks] {e}")
        conn.close()
        return

    batters = load_batter_features(conn, pick_date)
    if batters.empty:
        print(f"[picks] No batter stats found before {pick_date}. Run build_features.py first.")
        conn.close()
        return

    odds_df = load_live_odds(conn)

    # Score all batters
    batters["model_prob"] = batters.apply(lambda row: compute_hit_prob(row, params), axis=1)
    batters["p_hit"]      = batters["model_prob"]

    # Join with live odds on player name (fuzzy-safe: try exact match first)
    picks = batters.sort_values("model_prob", ascending=False).head(top_n * 3)

    # Merge odds if available
    if not odds_df.empty:
        picks = picks.merge(
            odds_df[["player", "line", "price", "bookmaker"]].rename(
                columns={"player": "batter_name", "price": "odds_over"}
            ),
            on="batter_name",
            how="left",
        )
    else:
        picks["line"]      = None
        picks["odds_over"] = None
        picks["bookmaker"] = None

    # Compute EV where odds are available
    picks["ev"] = picks.apply(
        lambda r: ev(r["model_prob"], american_to_decimal(int(r["odds_over"])))
        if pd.notna(r.get("odds_over")) else None,
        axis=1,
    )
    picks["value_bet"] = picks["ev"].apply(lambda x: 1 if pd.notna(x) and x > 0.05 else 0)

    # Final top N
    picks = picks.sort_values("model_prob", ascending=False).head(top_n)

    # Write to DB
    conn.execute("DELETE FROM daily_picks WHERE pick_date = ?", (pick_date,))
    for _, r in picks.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO daily_picks
                (pick_date, batter_id, batter_name, team, opponent, pitcher_name,
                 pitcher_throws, model_prob, expected_pa, p_hit,
                 odds_over, line, ev, value_bet)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            pick_date,
            int(r["batter_id"]) if pd.notna(r.get("batter_id")) else None,
            r.get("batter_name"),
            None, None, None, None,
            float(r["model_prob"]),
            float(r.get("pa", 3.5)),
            float(r["p_hit"]),
            int(r["odds_over"]) if pd.notna(r.get("odds_over")) else None,
            float(r["line"]) if pd.notna(r.get("line")) else None,
            float(r["ev"]) if pd.notna(r.get("ev")) else None,
            int(r["value_bet"]),
        ))
    conn.commit()
    conn.close()

    # Print summary
    print(f"\n[picks] Top {top_n} MLB hit picks for {pick_date}")
    print(f"{'Batter':<28} {'Prob':>6}  {'Line':>5}  {'Odds':>6}  {'EV':>7}  {'Value'}")
    print("-" * 65)
    for _, r in picks.iterrows():
        ev_str   = f"{r['ev']:.3f}" if pd.notna(r.get("ev")) else "   n/a"
        odds_str = str(int(r["odds_over"])) if pd.notna(r.get("odds_over")) else "  n/a"
        line_str = str(r["line"]) if pd.notna(r.get("line")) else " n/a"
        value    = "★" if r["value_bet"] else ""
        print(f"  {str(r.get('batter_name','')):<26} {r['model_prob']:>6.3f}  {line_str:>5}  {odds_str:>6}  {ev_str:>7}  {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--top",  type=int, default=10)
    args = parser.parse_args()
    run_picks(args.date, args.top)
