"""
nba/model/train_model.py — Train NBA player prop over/under classifiers.

Trains four calibrated HistGradientBoosting models:
  - points_model   : P(player goes OVER their points line)
  - rebounds_model : P(player goes OVER their rebounds line)
  - assists_model  : P(player goes OVER their assists line)
  - threes_model   : P(player goes OVER their 3PM line)

Common prop lines (used as training thresholds):
  - points: 15.5, 20.5, 25.5 (dynamic, based on player's season avg)
  - rebounds: 4.5, 6.5, 8.5
  - assists: 3.5, 5.5, 7.5
  - threes: 1.5, 2.5, 3.5

Models are saved as .pkl files in nba/models/.

Usage:
    python nba/model/train_model.py
    python nba/model/train_model.py --prop points
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.db import get_conn
from shared.config import RANDOM_STATE
from nba.features.build_features import load_stats, build_rolling_features, get_feature_cols

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Dynamic line thresholds per prop — model trains to predict P(stat > threshold)
PROP_THRESHOLDS = {
    "points":              lambda avg: round(avg - 0.5, 1) if avg else 15.5,
    "rebounds":            lambda avg: round(avg - 0.5, 1) if avg else 4.5,
    "assists":             lambda avg: round(avg - 0.5, 1) if avg else 3.5,
    "three_pointers_made": lambda avg: 1.5,
}

PROP_STAT_MAP = {
    "points":   "points",
    "rebounds": "rebounds",
    "assists":  "assists",
    "threes":   "three_pointers_made",
}


def make_label(df: pd.DataFrame, stat_col: str, threshold_fn) -> pd.Series:
    """Label = 1 if player exceeded their dynamic threshold."""
    avg = df.groupby("player_id")[stat_col].transform("mean")
    thresholds = avg.apply(threshold_fn)
    return (df[stat_col] > thresholds).astype(int)


def train_prop_model(df: pd.DataFrame, prop: str) -> dict:
    stat_col = PROP_STAT_MAP[prop]
    if stat_col not in df.columns:
        raise ValueError(f"Column {stat_col} not in features DataFrame")

    feature_cols = [c for c in get_feature_cols() if c in df.columns]

    df = df.dropna(subset=feature_cols + [stat_col])
    X = df[feature_cols]
    y = make_label(df, stat_col, PROP_THRESHOLDS[prop if prop != "threes" else "three_pointers_made"])

    print(f"[model] {prop}: {len(X)} samples, {y.mean():.3f} positive rate")

    # Chronological split — last 20% as validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", CalibratedClassifierCV(
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_depth=4,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
            ),
            cv=5,
            method="isotonic",
        )),
    ])

    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_val)[:, 1]
    brier = brier_score_loss(y_val, probs)
    auc   = roc_auc_score(y_val, probs) if y_val.nunique() > 1 else None

    print(f"[model] {prop}: Brier={brier:.4f}  AUC={auc:.4f if auc else 'n/a'}")

    # Save model
    model_path = MODELS_DIR / f"{prop}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "feature_cols": feature_cols}, f)
    print(f"[model] Saved → {model_path}")

    return {
        "prop": prop,
        "samples": len(X),
        "positive_rate": float(y.mean()),
        "brier_score": float(brier),
        "auc": float(auc) if auc else None,
        "feature_cols": feature_cols,
        "trained_at": datetime.now().isoformat(),
    }


def train_all(props: list[str]) -> dict:
    conn = get_conn("nba")
    df = load_stats(conn)
    conn.close()

    if df.empty:
        print("[model] No data found. Run nba/ingest/pull_player_stats.py first.")
        return {}

    print(f"[model] Loaded {len(df):,} player-game rows, building features ...")
    df = build_rolling_features(df)

    results = {}
    for prop in props:
        try:
            results[prop] = train_prop_model(df, prop)
        except Exception as e:
            print(f"[model] ERROR training {prop}: {e}")

    # Save report
    report_path = MODELS_DIR / "train_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[model] Report saved → {report_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prop", choices=["points", "rebounds", "assists", "threes", "all"],
                        default="all")
    args = parser.parse_args()

    props = list(PROP_STAT_MAP.keys()) if args.prop == "all" else [args.prop]
    train_all(props)
