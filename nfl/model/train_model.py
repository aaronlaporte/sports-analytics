"""
nfl/model/train_model.py — Train NFL anytime-TD and 2+TD classifiers.

Models:
  - anytime_td_model.pkl : CalibratedClassifier(LogisticRegression) — P(player scores any TD)
  - td2plus_model.pkl    : Pipeline(StandardScaler + PoissonRegressor) — P(player scores 2+ TD)
  - rec_count_model.pkl  : Pipeline(StandardScaler + Ridge) — expected receptions

Uses recency-weighted training (half-life = 45 days).

Usage:
    python nfl/model/train_model.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge, PoissonRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.db import get_conn
from shared.config import RANDOM_STATE
from nfl.features.build_training_table import build_training_table, FEATURE_COLS

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

DECAY_HALFLIFE_DAYS = 45

RECEPTION_FEATURES = [
    "targets_rolling_3", "targets_rolling_5", "targets_ewm_3", "targets_ewm_5",
    "receptions_rolling_3", "receptions_rolling_5", "receptions_ewm_3", "receptions_ewm_5",
    "touches_rolling_3", "touches_rolling_5",
    "total_td_rolling_3", "total_td_rolling_5",
    "season_td_rate",
    "points_scored_rolling_3", "points_scored_rolling_5",
    "opp_points_allowed_rolling_3", "opp_points_allowed_rolling_5",
    "is_home", "touch_share", "target_share",
    "pos_wr", "pos_rb", "pos_te",
]

TD2PLUS_FEATURES = [
    "touches_rolling_3", "touches_rolling_5", "touches_ewm_3", "touches_ewm_5",
    "targets_rolling_3", "targets_rolling_5",
    "total_td_rolling_3", "total_td_rolling_5", "total_td_ewm_3", "total_td_ewm_5",
    "season_td_rate",
    "points_scored_rolling_3", "points_scored_rolling_5",
    "opp_points_allowed_rolling_3", "opp_points_allowed_rolling_5",
    "opp_td_allowed_rolling_3", "opp_td_allowed_rolling_5",
    "td_allowed_pos_rolling_3", "td_allowed_pos_rolling_5",
    "touch_share", "target_share", "rush_share",
    "is_home", "denver_game",
]


def _recency_weights(dates: pd.Series) -> np.ndarray:
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days.clip(lower=0).astype(float)
    return (0.5 ** (days_ago / float(DECAY_HALFLIFE_DAYS))).to_numpy()


def time_split(df: pd.DataFrame):
    df = df.sort_values("game_date").reset_index(drop=True)
    idx = int(len(df) * 0.8)
    return df.iloc[:idx].copy(), df.iloc[idx:].copy()


def _fit_anytime(X_train, y_train, X_val, y_val, w):
    base = LogisticRegression(max_iter=200, class_weight="balanced",
                              solver="liblinear", random_state=RANDOM_STATE)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X_train, y_train, sample_weight=w)
    probs = model.predict_proba(X_val)[:, 1]
    metrics = {"brier_anytime": float(brier_score_loss(y_val, probs))}
    if y_val.nunique() > 1:
        metrics["auc_anytime"] = float(roc_auc_score(y_val, probs))
    return model, metrics


def _fit_td2plus(X_train, y_train, X_val, y_val, w):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("poisson", PoissonRegressor(alpha=0.001, max_iter=300)),
    ])
    model.fit(X_train, y_train, poisson__sample_weight=w)
    lam = model.predict(X_val).clip(min=0)
    p2 = 1 - np.exp(-lam) * (1 + lam)
    y_2plus = (y_val >= 2).astype(int)
    metrics = {"brier_td2plus": float(brier_score_loss(y_2plus, p2))}
    if y_2plus.nunique() > 1:
        metrics["auc_td2plus"] = float(roc_auc_score(y_2plus, p2))
    return model, metrics


def _fit_receptions(X_train, y_train, X_val, y_val, w):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])
    model.fit(X_train, y_train, ridge__sample_weight=w)
    preds = model.predict(X_val)
    return model, {"mae_receptions": float(mean_absolute_error(y_val, preds))}


def main():
    conn = get_conn("nfl")
    df = build_training_table(conn)
    conn.close()

    if df.empty:
        print("[nfl/model] No training data. Run ingest scripts first.")
        return

    # Filter to skill positions with enough data
    df = df[df["position"].isin(["RB", "WR", "TE", "QB"])].copy()
    df = df.dropna(subset=["game_date"])
    df["game_date"] = pd.to_datetime(df["game_date"])

    train_df, val_df = time_split(df)
    print(f"[nfl/model] Train: {len(train_df):,}  Val: {len(val_df):,}")

    w_train = _recency_weights(train_df["game_date"])

    # Ensure feature cols exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
            train_df[col] = 0.0
            val_df[col] = 0.0

    # Anytime TD
    anytime_model, m_any = _fit_anytime(
        train_df[FEATURE_COLS], train_df["anytime_td"].astype(int),
        val_df[FEATURE_COLS], val_df["anytime_td"].astype(int),
        w_train,
    )

    # Per-position anytime models
    anytime_by_pos = {}
    for pos in ["RB", "WR", "TE"]:
        sub_t = train_df[train_df["position"] == pos]
        sub_v = val_df[val_df["position"] == pos]
        if len(sub_t) >= 200 and len(sub_v) >= 50:
            w = _recency_weights(sub_t["game_date"])
            m, _ = _fit_anytime(
                sub_t[FEATURE_COLS], sub_t["anytime_td"].astype(int),
                sub_v[FEATURE_COLS], sub_v["anytime_td"].astype(int),
                w,
            )
            anytime_by_pos[pos] = m

    # 2+ TD
    td2_features = [c for c in TD2PLUS_FEATURES if c in train_df.columns]
    td2plus_model, m_2plus = _fit_td2plus(
        train_df[td2_features], train_df["total_tds"].fillna(0).astype(float),
        val_df[td2_features], val_df["total_tds"].fillna(0).astype(float),
        w_train,
    )

    # Receptions (WR/RB/TE only)
    rec_train = train_df[train_df["position"].isin(["WR", "RB", "TE"])]
    rec_val   = val_df[val_df["position"].isin(["WR", "RB", "TE"])]
    rec_features = [c for c in RECEPTION_FEATURES if c in rec_train.columns]
    w_rec = _recency_weights(rec_train["game_date"])
    rec_model, m_rec = _fit_receptions(
        rec_train[rec_features], rec_train["receptions"].fillna(0).astype(float),
        rec_val[rec_features], rec_val["receptions"].fillna(0).astype(float),
        w_rec,
    )

    # Stacker: blend global + position anytime probs
    global_any_val = anytime_model.predict_proba(val_df[FEATURE_COLS])[:, 1]
    pos_any_val = np.zeros(len(val_df))
    for pos, sub in val_df.groupby("position"):
        m = anytime_by_pos.get(pos, anytime_model)
        pos_any_val[sub.index] = m.predict_proba(sub[FEATURE_COLS])[:, 1]
    stack_X = np.vstack([global_any_val, pos_any_val]).T
    stacker = LogisticRegression(max_iter=200, solver="liblinear")
    stacker.fit(stack_X, val_df["anytime_td"].astype(int))

    # TD2+ stacker
    global_lam = td2plus_model.predict(val_df[td2_features]).clip(0)
    global_p2  = 1 - np.exp(-global_lam) * (1 + global_lam)
    td2_stacker = LogisticRegression(max_iter=200, solver="liblinear")
    td2_stacker.fit(global_p2.reshape(-1, 1), (val_df["total_tds"].fillna(0) >= 2).astype(int))

    metrics = {}
    metrics.update(m_any)
    metrics.update(m_2plus)
    metrics.update(m_rec)
    print(f"[nfl/model] Metrics: {metrics}")

    dump({
        "model": anytime_model,
        "by_position": anytime_by_pos,
        "stacker": stacker,
        "features": FEATURE_COLS,
    }, MODELS_DIR / "anytime_td_model.pkl")

    dump({
        "model": td2plus_model,
        "stacker": td2_stacker,
        "features": td2_features,
    }, MODELS_DIR / "td2plus_model.pkl")

    dump({
        "model": rec_model,
        "features": rec_features,
    }, MODELS_DIR / "rec_count_model.pkl")

    report = {
        "trained_at": datetime.now().isoformat(),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "metrics": metrics,
    }
    with open(MODELS_DIR / "train_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[nfl/model] Saved models → {MODELS_DIR}")


if __name__ == "__main__":
    main()
