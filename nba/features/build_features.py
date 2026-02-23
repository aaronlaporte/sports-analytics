"""
nba/features/build_features.py — Build rolling feature table for NBA player props.

Reads player_game_stats from nba.db, computes rolling windows and contextual
features, and returns a DataFrame ready for model training or prediction.

Usage:
    python nba/features/build_features.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.db import get_conn
from shared.config import ROLLING_WINDOWS

STAT_COLS = [
    "points", "rebounds", "assists", "three_pointers_made",
    "steals", "blocks", "turnovers", "minutes",
    "field_goals_att", "free_throws_att",
    "usage_rate", "true_shooting", "player_efficiency", "fantasy_pts_dk",
]

MIN_MINUTES = 10   # filter out DNPs


def load_stats(conn) -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT * FROM player_game_stats
        WHERE is_game_over = 1
          AND minutes >= ?
        ORDER BY player_id, game_date
    """, conn, params=(MIN_MINUTES,))
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "game_date"]).copy()

    for col in STAT_COLS:
        if col not in df.columns:
            continue
        # Shift by 1 so we never leak the current game's stats
        shifted = df.groupby("player_id")[col].shift(1)
        for w in ROLLING_WINDOWS:
            df[f"{col}_rolling_{w}"] = (
                shifted.groupby(df["player_id"])
                .transform(lambda s: s.rolling(w, min_periods=1).mean())
            )
        # EWM span=5 (recent form)
        df[f"{col}_ewm5"] = (
            shifted.groupby(df["player_id"])
            .transform(lambda s: s.ewm(span=5, adjust=False).mean())
        )

    # Season averages (prior games, cumulative)
    for col in ["points", "rebounds", "assists", "three_pointers_made"]:
        if col not in df.columns:
            continue
        cum = df.groupby("player_id")[col].cumsum() - df[col]
        cnt = df.groupby("player_id").cumcount()
        df[f"{col}_season_avg"] = (cum / cnt.replace(0, np.nan)).round(2)

    # Days rest
    df["prev_game_date"] = df.groupby("player_id")["game_date"].shift(1)
    df["days_rest"] = (df["game_date"] - df["prev_game_date"]).dt.days.fillna(3).clip(0, 14)

    # Home/away
    df["is_home"] = (df["home_or_away"] == "HOME").astype(int)

    # Injury flags
    df["injury_questionable"] = (df["injury_status"] == "Questionable").astype(int)
    df["injury_doubtful"]     = (df["injury_status"] == "Doubtful").astype(int)

    # Opponent difficulty (lower rank = tougher defense)
    df["opp_rank_pts"]    = df["opponent_rank"].fillna(15)
    df["opp_rank_pos"]    = df["opp_position_rank"].fillna(15)
    df["tough_defense"]   = (df["opp_rank_pts"] <= 10).astype(int)

    # DK salary as proxy for expected role/minutes
    df["salary_dk_norm"] = (df["salary_dk"].fillna(0) / 10000).clip(0, 1)

    # Position one-hot (from player_game_stats if available, else skip)

    return df.drop(columns=["prev_game_date"], errors="ignore")


def get_feature_cols() -> list[str]:
    """Return the list of feature columns used for modeling."""
    cols = []
    for col in STAT_COLS:
        for w in ROLLING_WINDOWS:
            cols.append(f"{col}_rolling_{w}")
        cols.append(f"{col}_ewm5")
    for col in ["points", "rebounds", "assists", "three_pointers_made"]:
        cols.append(f"{col}_season_avg")
    cols += [
        "days_rest", "is_home",
        "injury_questionable", "injury_doubtful",
        "opp_rank_pts", "opp_rank_pos", "tough_defense",
        "salary_dk_norm",
    ]
    return cols


if __name__ == "__main__":
    conn = get_conn("nba")
    df = load_stats(conn)
    conn.close()
    print(f"Loaded {len(df):,} player-game rows")
    df = build_rolling_features(df)
    print(f"Feature matrix shape: {df.shape}")
    print("Feature cols:", get_feature_cols()[:10], "...")
