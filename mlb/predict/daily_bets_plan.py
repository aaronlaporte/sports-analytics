"""
mlb/predict/daily_bets_plan.py — Generate a 4-bet daily MLB plan:
  - 2 singles (1+ hit)
  - 1 parlay (same 2 batters)
  - 1 HR pick

Reads model params from DB (or mlb/models/hit_prob_model.json),
scores all qualified batters, joins live odds from live_odds table.

Usage:
    python mlb/predict/daily_bets_plan.py
    python mlb/predict/daily_bets_plan.py --as-of 2025-04-15
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.db import get_conn

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "hit_prob_model.json"

MIN_EXPECTED_PA = 4.0
MIN_RECENT_PA   = 40.0
MIN_HIT_PROB    = 0.65


def _shrink_rate(hits, pa, league_rate, prior_pa):
    return (hits + league_rate * prior_pa) / (pa + prior_pa)


def _recent_pa_window(group, window_pa: int) -> pd.DataFrame:
    pa = group["PA"].to_numpy(dtype=float)
    hits = group["H"].to_numpy(dtype=float)
    hrs = group.get("HR", pd.Series(0, index=group.index)).to_numpy(dtype=float)
    cum_pa = np.cumsum(pa)
    cum_hits = np.cumsum(hits)
    cum_hrs = np.cumsum(hrs)
    cum_pa_prior = np.concatenate([[0.0], cum_pa[:-1]])
    cum_hits_prior = np.concatenate([[0.0], cum_hits[:-1]])
    cum_hrs_prior = np.concatenate([[0.0], cum_hrs[:-1]])
    left = np.searchsorted(cum_pa_prior, cum_pa_prior - window_pa, side="left")
    pa_left = np.where(left > 0, cum_pa_prior[left - 1], 0.0)
    hits_left = np.where(left > 0, cum_hits_prior[left - 1], 0.0)
    hrs_left = np.where(left > 0, cum_hrs_prior[left - 1], 0.0)
    return pd.DataFrame(
        {
            f"recent_{window_pa}_pa": cum_pa_prior - pa_left,
            f"recent_{window_pa}_hits": cum_hits_prior - hits_left,
            f"recent_{window_pa}_hrs": cum_hrs_prior - hrs_left,
        },
        index=group.index,
    )


def load_model_params() -> dict:
    if MODEL_PATH.exists():
        with open(MODEL_PATH) as f:
            return json.load(f)
    # Fallback: load from DB
    conn = get_conn("mlb")
    row = conn.execute(
        "SELECT params_json FROM model_params ORDER BY run_date DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row:
        return json.loads(row["params_json"])
    raise FileNotFoundError(
        "No model params found. Run mlb/model/train_model.py first."
    )


def score_batters(as_of: date = None) -> pd.DataFrame:
    params = load_model_params()
    conn = get_conn("mlb")
    df = pd.read_sql(
        "SELECT game_date, batter_id, pa, hits AS H, hr, season_pa FROM batter_stats "
        "ORDER BY batter_id, game_date",
        conn,
    )
    conn.close()

    df["game_date"] = pd.to_datetime(df["game_date"])
    if as_of:
        df = df[df["game_date"] <= pd.Timestamp(as_of)]

    # Rename for compatibility
    df = df.rename(columns={"pa": "PA"})
    if "hr" not in df.columns:
        df["hr"] = 0

    group = df.groupby("batter_id", group_keys=False)
    df["season_hits_prior"] = group["H"].cumsum().shift(1)
    df["season_pa_prior"] = group["PA"].cumsum().shift(1)
    df["season_hr_prior"] = group["hr"].cumsum().shift(1)
    df["season_ab_prior"] = (group["PA"].cumsum().shift(1) * 0.85).round()  # AB ≈ 0.85 * PA
    df["games_prior"] = group.cumcount()

    recent_50 = group.apply(_recent_pa_window, window_pa=50).reset_index(level=0, drop=True)
    df = pd.concat([df, recent_50], axis=1)

    df["pa_per_game_prior"] = df["season_pa_prior"] / df["games_prior"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["season_hits_prior", "season_pa_prior", "pa_per_game_prior"])
    df = df[
        (df["season_pa_prior"] >= params["min_prior_pa"])
        & (df["games_prior"] >= params["min_prior_games"])
        & (df["recent_50_pa"] >= params["min_recent_pa"])
    ]

    league_rate = params["league_hit_per_pa"]

    season_rate = _shrink_rate(df["season_hits_prior"], df["season_pa_prior"],
                               league_rate, params["season_prior_pa"])
    recent_rate = _shrink_rate(df["recent_50_hits"], df["recent_50_pa"],
                               league_rate, params["recent_prior_pa"])
    base_weight = max(0.0, 1 - params["recent_weight"] - params["split_weight"])
    p_hit_per_pa = base_weight * season_rate + params["recent_weight"] * recent_rate

    expected_pa = df["pa_per_game_prior"].clip(params["pa_min"], params["pa_max"])
    df["p_at_least_1_hit"] = 1 - np.power(1 - p_hit_per_pa, expected_pa)
    df["expected_pa"] = expected_pa

    # HR model: shrink HR per AB
    league_hr_rate = df["season_hr_prior"].sum() / max(df["season_ab_prior"].sum(), 1)
    hr_rate = _shrink_rate(
        df["season_hr_prior"].fillna(0),
        df["season_ab_prior"].fillna(0),
        league_hr_rate,
        200,
    )
    ab_per_pa = (df["season_ab_prior"] / df["season_pa_prior"]).fillna(0.7).clip(0.55, 0.9)
    expected_ab = (expected_pa * ab_per_pa).clip(2.5, 4.5)
    df["p_at_least_1_hr"] = 1 - np.power(1 - hr_rate, expected_ab)

    # Latest row per batter (most recent game's cumulative stats)
    latest = (
        df.sort_values(["batter_id", "game_date"])
        .groupby("batter_id", as_index=False)
        .tail(1)
    )
    return latest


def build_plan(as_of: date = None):
    latest = score_batters(as_of)

    # Hit pool
    hit_pool = latest[
        (latest["expected_pa"] >= MIN_EXPECTED_PA)
        & (latest["recent_50_pa"] >= MIN_RECENT_PA)
        & (latest["p_at_least_1_hit"] >= MIN_HIT_PROB)
    ].copy().sort_values("p_at_least_1_hit", ascending=False)

    hit_singles = hit_pool.head(2)
    parlay = hit_singles.copy()

    hr_pick = latest.sort_values("p_at_least_1_hr", ascending=False).head(1).copy()

    # Try to join live odds
    conn = get_conn("mlb")
    try:
        odds_df = pd.read_sql("""
            SELECT player, name, line, price, market
            FROM live_odds
            WHERE market = 'batter_hits'
              AND name = 'Over'
            ORDER BY fetched_at DESC
        """, conn)
    except Exception:
        odds_df = pd.DataFrame()
    conn.close()

    bets = []

    def fmt(df_part, bet_type):
        rows = []
        for _, r in df_part.iterrows():
            line, odds_val = None, None
            if not odds_df.empty:
                match = odds_df[odds_df["player"].str.lower().str.contains(
                    str(r.get("batter_id", "")), na=False
                )]
                if not match.empty:
                    line = match.iloc[0]["line"]
                    odds_val = match.iloc[0]["price"]
            prob = r["p_at_least_1_hit"] if bet_type != "single_hr" else r["p_at_least_1_hr"]
            rows.append({
                "game_date": r["game_date"].date() if hasattr(r["game_date"], "date") else r["game_date"],
                "batter_id": r["batter_id"],
                "expected_pa": round(r["expected_pa"], 2),
                "p_at_least_1_hit": round(prob, 4),
                "bet_type": bet_type,
                "odds": odds_val,
                "line": line,
            })
        return pd.DataFrame(rows)

    if not hit_singles.empty:
        bets.append(fmt(hit_singles, "single_hit"))
        bets.append(fmt(parlay, "parlay_hit"))
    if not hr_pick.empty:
        bets.append(fmt(hr_pick, "single_hr"))

    if bets:
        output = pd.concat(bets, ignore_index=True)
    else:
        output = pd.DataFrame()

    print(f"\n=== MLB Daily Bets Plan ({as_of or 'latest'}) ===")
    if output.empty:
        print("No qualifying picks found.")
    else:
        print(output.to_string(index=False))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=None)
    args = parser.parse_args()
    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    build_plan(as_of)
