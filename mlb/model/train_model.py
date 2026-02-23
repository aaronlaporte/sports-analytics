"""
mlb/model/train_model.py — Train the MLB hit-probability beta-binomial model.

Model: shrinkage blend of season rate, recent-50-PA rate, and pitcher-hand split rate.
Uses grid search over prior_pa and blend weights; picks minimum Brier on validation.

Output: mlb/models/hit_prob_model.json — model params used by daily_picks.py

Usage:
    python mlb/model/train_model.py
"""

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.db import get_conn

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class HitProbModelParams:
    league_hit_per_pa: float
    season_prior_pa: int
    recent_prior_pa: int
    recent_weight: float
    split_prior_pa: int
    split_weight: float
    min_prior_pa: int
    min_prior_games: int
    min_recent_pa: int
    pa_min: float
    pa_max: float


def _shrink_rate(hits: pd.Series, pa: pd.Series, league_rate: float, prior_pa: int) -> pd.Series:
    return (hits + league_rate * prior_pa) / (pa + prior_pa)


def _recent_pa_window(group: pd.DataFrame, window_pa: int) -> pd.DataFrame:
    pa = group["pa"].to_numpy(dtype=float)
    hits = group["hits"].to_numpy(dtype=float)
    cum_pa = np.cumsum(pa)
    cum_hits = np.cumsum(hits)
    cum_pa_prior = np.concatenate([[0.0], cum_pa[:-1]])
    cum_hits_prior = np.concatenate([[0.0], cum_hits[:-1]])
    left = np.searchsorted(cum_pa_prior, cum_pa_prior - window_pa, side="left")
    pa_left = np.where(left > 0, cum_pa_prior[left - 1], 0.0)
    hits_left = np.where(left > 0, cum_hits_prior[left - 1], 0.0)
    return pd.DataFrame(
        {
            f"recent_{window_pa}_pa": cum_pa_prior - pa_left,
            f"recent_{window_pa}_hits": cum_hits_prior - hits_left,
        },
        index=group.index,
    )


def load_batter_stats(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM batter_stats ORDER BY batter_id, game_date", conn)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def build_feature_frame(
    df: pd.DataFrame,
    min_prior_pa: int,
    min_prior_games: int,
    min_recent_pa: int,
) -> pd.DataFrame:
    # Rename to match original model column names
    df = df.rename(columns={"hits": "H", "pa": "PA"})
    df = df.sort_values(["batter_id", "game_date"]).copy()

    group = df.groupby("batter_id", group_keys=False)
    df["season_hits_prior"] = group["H"].cumsum().shift(1)
    df["season_pa_prior"] = group["PA"].cumsum().shift(1)
    df["games_prior"] = group.cumcount()

    recent_50 = group.apply(_recent_pa_window, window_pa=50).reset_index(level=0, drop=True)
    df = pd.concat([df, recent_50], axis=1)

    df["pa_per_game_prior"] = df["season_pa_prior"] / df["games_prior"]
    df["label_hit"] = (df["H"] >= 1).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["season_hits_prior", "season_pa_prior", "pa_per_game_prior"])
    df = df[
        (df["season_pa_prior"] >= min_prior_pa)
        & (df["games_prior"] >= min_prior_games)
        & (df["recent_50_pa"] >= min_recent_pa)
    ]
    # No hand-split data in statcast raw (pitcher hand lookups are separate)
    df["split_hits_prior_R"] = 0.0
    df["split_pa_prior_R"] = 0.0
    df["split_hits_prior_L"] = 0.0
    df["split_pa_prior_L"] = 0.0
    df["last_pitcher_hand"] = None
    return df


def score_predictions(
    df: pd.DataFrame,
    league_rate: float,
    season_prior_pa: int,
    recent_prior_pa: int,
    recent_weight: float,
    split_prior_pa: int,
    split_weight: float,
    pa_min: float,
    pa_max: float,
) -> pd.Series:
    season_rate = _shrink_rate(df["season_hits_prior"], df["season_pa_prior"], league_rate, season_prior_pa)
    recent_rate = _shrink_rate(df["recent_50_hits"], df["recent_50_pa"], league_rate, recent_prior_pa)
    # Without reliable split data fall back to season rate for splits
    split_rate = season_rate
    base_weight = max(0.0, 1 - recent_weight - split_weight)
    p_hit_per_pa = base_weight * season_rate + recent_weight * recent_rate + split_weight * split_rate
    expected_pa = df["pa_per_game_prior"].clip(pa_min, pa_max)
    return 1 - np.power(1 - p_hit_per_pa, expected_pa)


def brier_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def train_model(
    df: pd.DataFrame,
    season_prior_grid: Iterable[int],
    recent_weight_grid: Iterable[float],
    pa_min: float,
    pa_max: float,
    min_prior_pa: int,
    min_prior_games: int,
    min_recent_pa: int,
) -> Tuple[HitProbModelParams, float]:
    df = df.sort_values("game_date")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    valid_df = df.iloc[split_idx:]

    league_rate = train_df["season_hits_prior"].sum() / max(train_df["season_pa_prior"].sum(), 1)

    best_score = float("inf")
    best_params = None

    for season_prior_pa in season_prior_grid:
        for recent_weight in recent_weight_grid:
            preds = score_predictions(
                valid_df,
                league_rate,
                season_prior_pa,
                recent_prior_pa=50,
                recent_weight=recent_weight,
                split_prior_pa=50,
                split_weight=0.0,
                pa_min=pa_min,
                pa_max=pa_max,
            )
            score = brier_score(valid_df["label_hit"], preds)
            if score < best_score:
                best_score = score
                best_params = HitProbModelParams(
                    league_hit_per_pa=float(league_rate),
                    season_prior_pa=int(season_prior_pa),
                    recent_prior_pa=50,
                    recent_weight=float(recent_weight),
                    split_prior_pa=50,
                    split_weight=0.0,
                    min_prior_pa=int(min_prior_pa),
                    min_prior_games=int(min_prior_games),
                    min_recent_pa=int(min_recent_pa),
                    pa_min=float(pa_min),
                    pa_max=float(pa_max),
                )

    if best_params is None:
        raise RuntimeError("No parameter combinations evaluated.")
    return best_params, best_score


def main():
    conn = get_conn("mlb")
    df = load_batter_stats(conn)
    conn.close()

    if df.empty:
        print("[mlb/model] No batter_stats found. Run mlb/features/build_features.py first.")
        return

    print(f"[mlb/model] Loaded {len(df):,} batter-game rows")

    feature_df = build_feature_frame(df, min_prior_pa=20, min_prior_games=5, min_recent_pa=30)
    print(f"[mlb/model] Feature frame: {len(feature_df):,} qualifying rows")

    params, brier = train_model(
        feature_df,
        season_prior_grid=[50, 100, 200, 400],
        recent_weight_grid=[0.0, 0.2, 0.4],
        pa_min=3.0,
        pa_max=5.0,
        min_prior_pa=20,
        min_prior_games=5,
        min_recent_pa=30,
    )

    model_path = MODELS_DIR / "hit_prob_model.json"
    payload = asdict(params)
    payload["validation_brier"] = brier
    payload["trained_at"] = datetime.now().isoformat()

    with open(model_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Also write to DB for reference
    conn2 = get_conn("mlb")
    conn2.execute("""
        INSERT OR REPLACE INTO model_params
            (run_date, league_hit_per_pa, season_prior_pa, recent_weight, split_weight, brier_score, params_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().date().isoformat(),
        params.league_hit_per_pa,
        params.season_prior_pa,
        params.recent_weight,
        params.split_weight,
        brier,
        json.dumps(payload),
    ))
    conn2.commit()
    conn2.close()

    print(f"[mlb/model] Validation Brier={brier:.4f}")
    print(f"[mlb/model] Saved → {model_path}")


if __name__ == "__main__":
    main()
