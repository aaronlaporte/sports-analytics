"""
nba/predict/daily_picks.py — Score tonight's NBA player props with trained models.

Loads trained models from nba/models/, builds features for tonight's players,
joins live odds from live_odds table, computes EV, and writes to daily_picks.

Usage:
    python nba/predict/daily_picks.py
    python nba/predict/daily_picks.py --min-ev 0.05 --top 20
"""

import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.db import get_conn, create_schema
from nba.features.build_features import load_stats, build_rolling_features, get_feature_cols

MODELS_DIR = Path(__file__).parent.parent / "models"

PROP_MARKET_MAP = {
    "points":   "player_points",
    "rebounds": "player_rebounds",
    "assists":  "player_assists",
    "threes":   "player_threes",
}


def load_model(prop: str) -> dict:
    path = MODELS_DIR / f"{prop}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run nba/model/train_model.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def compute_ev(prob: float, decimal_odds: float) -> float:
    return prob * decimal_odds - 1


def get_todays_players(conn) -> pd.DataFrame:
    """Return players in today's games from player_game_stats (is_game_over=0)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df = pd.read_sql("""
        SELECT * FROM player_game_stats
        WHERE game_date = ?
    """, conn, params=(today,))
    return df


def score_tonight(min_ev: float = 0.0, top: int = 30):
    create_schema("nba")
    conn = get_conn("nba")

    # Build features on full history
    hist = load_stats(conn)
    if hist.empty:
        print("[nba/picks] No stats data. Run nba/ingest/pull_player_stats.py first.")
        conn.close()
        return

    hist = build_rolling_features(hist)
    feature_cols = get_feature_cols()

    # Load live odds
    try:
        odds_df = pd.read_sql("""
            SELECT player, market, name, line, price
            FROM live_odds
            ORDER BY fetched_at DESC
        """, conn)
    except Exception:
        odds_df = pd.DataFrame()

    pick_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    all_picks = []

    for prop, market in PROP_MARKET_MAP.items():
        try:
            bundle = load_model(prop)
        except FileNotFoundError as e:
            print(f"[nba/picks] WARN: {e}")
            continue

        clf = bundle["model"]
        feat_cols = [c for c in bundle.get("feature_cols", feature_cols) if c in hist.columns]

        # Latest feature row per player (most recent game)
        latest = (
            hist.dropna(subset=feat_cols)
            .sort_values(["player_id", "game_date"])
            .groupby("player_id", as_index=False)
            .tail(1)
        )
        if latest.empty:
            continue

        X = latest[feat_cols]
        probs = clf.predict_proba(X)[:, 1]  # P(over)
        latest = latest.copy()
        latest["p_over"] = probs

        # Join odds
        if not odds_df.empty:
            prop_odds = odds_df[
                (odds_df["market"] == market)
                & (odds_df["name"] == "Over")
            ].copy()
            # Find best odds per player
            if not prop_odds.empty:
                best_odds = (
                    prop_odds.sort_values("price", ascending=False)
                    .groupby("player")
                    .head(1)
                )
                # Attempt fuzzy name join — match by last name (simplified)
                best_odds = best_odds.copy()
                best_odds["last_name"] = best_odds["player"].str.split().str[-1].str.lower()
                latest["last_name"] = latest["player_name"].str.split().str[-1].str.lower()
                merged = latest.merge(
                    best_odds[["last_name", "line", "price"]].rename(
                        columns={"price": "best_odds", "line": "prop_line"}
                    ),
                    on="last_name",
                    how="left",
                )
            else:
                merged = latest.copy()
                merged["best_odds"] = None
                merged["prop_line"] = None
        else:
            merged = latest.copy()
            merged["best_odds"] = None
            merged["prop_line"] = None

        for _, r in merged.iterrows():
            prob = float(r["p_over"])
            best_odds_val = r.get("best_odds")
            prop_line = r.get("prop_line")
            ev = None
            value_bet = 0
            if pd.notna(best_odds_val) and best_odds_val is not None:
                dec = american_to_decimal(int(best_odds_val))
                ev = round(compute_ev(prob, dec), 4)
                value_bet = int(ev >= min_ev)

            all_picks.append({
                "pick_date":   pick_date,
                "player_id":   r["player_id"],
                "player_name": r["player_name"],
                "team":        r.get("team", ""),
                "opponent":    r.get("opponent", ""),
                "position":    r.get("position", ""),
                "prop_type":   prop,
                "model_prob":  round(prob, 4),
                "direction":   "Over",
                "line":        prop_line,
                "best_odds":   int(best_odds_val) if pd.notna(best_odds_val) and best_odds_val is not None else None,
                "best_book":   None,
                "ev":          ev,
                "value_bet":   value_bet,
            })

    picks_df = pd.DataFrame(all_picks)
    if picks_df.empty:
        print("[nba/picks] No picks generated.")
        conn.close()
        return

    # Save to DB
    picks_df.to_sql("daily_picks", conn, if_exists="append", index=False,
                    method="replace" if False else None)
    conn.close()

    # Display top picks by EV or model_prob
    display = picks_df[picks_df["value_bet"] == 1] if picks_df["value_bet"].any() else picks_df
    display = display.sort_values("model_prob", ascending=False).head(top)

    print(f"\n=== NBA Daily Picks — {pick_date} ===")
    print(display[[
        "player_name", "prop_type", "line", "model_prob", "best_odds", "ev", "value_bet"
    ]].to_string(index=False))

    return picks_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-ev", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    score_tonight(min_ev=args.min_ev, top=args.top)
