"""
mlb_insights/signal_engine/hr_features_v2.py -- HR prediction v2 sub-model.

Extends the base 7-feature HR model with 3 additional features:
    8. Fly ball rate (rolling 50 BBE)
    9. Hard-hit fly rate (fly balls with launch_speed >= 95 mph)
   10. Pitcher recent HR vulnerability (last 5 starts only)

The final p_hr_v2 is computed via a 10-feature logistic model.
"""

import json
import logging
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_insights.config import (
    DATA_DIR,
    DB_PATH,
    HR_FLY_BALL_MIN_BBE,
    HR_HARD_HIT_FLY_MIN_FLIES,
    HR_HARD_HIT_FLY_THRESHOLD,
    HR_PITCHER_RECENT_MIN_STARTS,
    HR_PITCHER_RECENT_STARTS,
    LEAGUE_AVG_FLY_BALL_RATE,
    LEAGUE_AVG_HARD_HIT_FLY_RATE,
)
from mlb_insights.signal_engine.hr_model import (
    ROLLING_BBE_WINDOW,
    compute_hr_features,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

HR_V2_INTERCEPT = -3.25
HR_V2_WEIGHTS = {
    "barrel_rate":            3.5,
    "avg_exit_velo":          0.06,
    "hr_pa_rate":             8.0,
    "pitcher_hr_vuln":        1.5,      # reduced since we now have pitcher_recent_hr_vuln
    "park_factor":            1.2,
    "platoon_hr_rate":        5.0,
    "power_trend":            1.5,
    "fly_ball_rate":          2.0,      # NEW
    "hard_hit_fly_rate":      3.0,      # NEW
    "pitcher_recent_hr_vuln": 2.0,      # NEW
}

_V2_FEATURE_NAMES = [
    "barrel_rate", "avg_exit_velo", "hr_pa_rate",
    "pitcher_hr_vuln", "park_factor", "platoon_hr_rate", "power_trend",
    "fly_ball_rate", "hard_hit_fly_rate", "pitcher_recent_hr_vuln",
]

# Feature centering baselines (includes new features)
_V2_FEATURE_BASELINES = {
    "barrel_rate":            0.065,
    "avg_exit_velo":          88.0,
    "hr_pa_rate":             0.035,
    "pitcher_hr_vuln":        1.0,
    "park_factor":            1.0,
    "platoon_hr_rate":        0.035,
    "power_trend":            0.0,
    "fly_ball_rate":          LEAGUE_AVG_FLY_BALL_RATE,       # 0.35
    "hard_hit_fly_rate":      LEAGUE_AVG_HARD_HIT_FLY_RATE,   # 0.25
    "pitcher_recent_hr_vuln": 1.0,
}

HR_V2_COEFFICIENTS_PATH = DATA_DIR / "hr_model_v2_coefficients.json"


# ── Coefficient Persistence ─────────────────────────────────────────────────

def load_hr_v2_coefficients() -> dict | None:
    """Load fitted HR v2 model coefficients from disk.

    Returns:
        Dict with keys 'intercept' and 'weights' (dict of feature->coeff),
        or None if the file does not exist or is malformed.
    """
    if not HR_V2_COEFFICIENTS_PATH.exists():
        return None
    try:
        with open(HR_V2_COEFFICIENTS_PATH) as f:
            data = json.load(f)
        if "intercept" not in data or "weights" not in data:
            logger.warning("hr_model_v2_coefficients.json missing required keys.")
            return None
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load HR v2 coefficients: %s", exc)
        return None


def fit_hr_model_v2(conn: sqlite3.Connection, min_samples: int = 500) -> dict | None:
    """Fit logistic regression on historical HR v2 features vs actual outcomes.

    Pulls rows from hr_features_v2 joined with batter_stats to determine
    whether the batter actually hit a HR on that date.  Centers features the
    same way the hardcoded model does, then fits sklearn LogisticRegression.

    Args:
        conn: Open sqlite3 connection.
        min_samples: Minimum rows required to fit.  Returns None if fewer.

    Returns:
        Dict with 'intercept' and 'weights', or None if insufficient data.
    """
    from sklearn.linear_model import LogisticRegression

    df = pd.read_sql_query("""
        SELECT
            hf.feature_date,
            hf.batter_id,
            hf.barrel_rate,
            hf.avg_exit_velo,
            hf.hr_pa_rate,
            hf.pitcher_hr_vuln,
            hf.park_factor,
            hf.platoon_hr_rate,
            hf.power_trend,
            hf.fly_ball_rate,
            hf.hard_hit_fly_rate,
            hf.pitcher_recent_hr_vuln,
            COALESCE(bs.home_runs, 0) AS actual_hr
        FROM hr_features_v2 hf
        LEFT JOIN batter_stats bs
            ON hf.batter_id = bs.batter_id
           AND hf.feature_date = bs.game_date
    """, conn)

    if len(df) < min_samples:
        logger.warning(
            "fit_hr_model_v2: only %d rows (need %d). Skipping refit.",
            len(df), min_samples,
        )
        return None

    y = (df["actual_hr"] >= 1).astype(int).values

    X = np.column_stack([
        df["barrel_rate"].values - _V2_FEATURE_BASELINES["barrel_rate"],
        df["avg_exit_velo"].values - _V2_FEATURE_BASELINES["avg_exit_velo"],
        df["hr_pa_rate"].values - _V2_FEATURE_BASELINES["hr_pa_rate"],
        df["pitcher_hr_vuln"].values - _V2_FEATURE_BASELINES["pitcher_hr_vuln"],
        np.log(np.clip(df["park_factor"].values, 0.5, None)),
        df["platoon_hr_rate"].values - _V2_FEATURE_BASELINES["platoon_hr_rate"],
        df["power_trend"].values - _V2_FEATURE_BASELINES["power_trend"],
        df["fly_ball_rate"].values - _V2_FEATURE_BASELINES["fly_ball_rate"],
        df["hard_hit_fly_rate"].values - _V2_FEATURE_BASELINES["hard_hit_fly_rate"],
        df["pitcher_recent_hr_vuln"].values - _V2_FEATURE_BASELINES["pitcher_recent_hr_vuln"],
    ])

    logger.info(
        "fit_hr_model_v2: fitting on %d samples (%d positive, %.2f%% HR rate).",
        len(y), y.sum(), 100.0 * y.mean(),
    )

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(X, y)

    coefficients = {
        "intercept": round(float(model.intercept_[0]), 6),
        "weights": {
            name: round(float(model.coef_[0][i]), 6)
            for i, name in enumerate(_V2_FEATURE_NAMES)
        },
    }

    HR_V2_COEFFICIENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HR_V2_COEFFICIENTS_PATH, "w") as f:
        json.dump(coefficients, f, indent=2)
    logger.info("Saved fitted HR v2 coefficients to %s", HR_V2_COEFFICIENTS_PATH)

    return coefficients


# ── V2 Feature Computation ──────────────────────────────────────────────────

def _compute_fly_ball_features(
    conn: sqlite3.Connection,
    game_date: str,
    batter_ids: list[int],
) -> dict[int, dict]:
    """Compute fly_ball_rate and hard_hit_fly_rate for each batter.

    Uses a rolling window of the last ROLLING_BBE_WINDOW batted ball events
    (same window as barrel_rate in v1).

    Args:
        conn: Open sqlite3 connection.
        game_date: Evaluation date (YYYY-MM-DD).
        batter_ids: List of batter MLBAM IDs.

    Returns:
        Dict of batter_id -> {"fly_ball_rate": float, "hard_hit_fly_rate": float}.
    """
    if not batter_ids:
        return {}

    placeholders = ",".join("?" * len(batter_ids))
    cur = conn.cursor()

    # Pull all BBEs for these batters before game_date, ordered for rolling window
    cur.execute(f"""
        SELECT batter, bb_type, launch_speed
        FROM statcast_staging
        WHERE batter IN ({placeholders})
          AND game_date < ?
          AND launch_speed IS NOT NULL
          AND launch_speed > 0
        ORDER BY batter, game_date, at_bat_number, pitch_number
    """, batter_ids + [game_date])

    # Organize by batter
    batter_bbes = defaultdict(list)
    for row in cur.fetchall():
        batter_bbes[row["batter"]].append({
            "bb_type": row["bb_type"],
            "launch_speed": row["launch_speed"],
        })

    results = {}
    for bid in batter_ids:
        bbes = batter_bbes.get(bid, [])
        recent = bbes[-ROLLING_BBE_WINDOW:]

        # ── Fly ball rate ────────────────────────────────────────────────
        if len(recent) >= HR_FLY_BALL_MIN_BBE:
            fly_balls = sum(1 for b in recent if b["bb_type"] == "fly_ball")
            fb_rate = fly_balls / len(recent)
        else:
            fb_rate = LEAGUE_AVG_FLY_BALL_RATE
            fly_balls = 0

        # ── Hard-hit fly rate ────────────────────────────────────────────
        fly_ball_events = [
            b for b in recent if b["bb_type"] == "fly_ball"
        ]
        if len(fly_ball_events) >= HR_HARD_HIT_FLY_MIN_FLIES:
            hard_hits = sum(
                1 for b in fly_ball_events
                if b["launch_speed"] >= HR_HARD_HIT_FLY_THRESHOLD
            )
            hh_fly_rate = hard_hits / len(fly_ball_events)
        else:
            hh_fly_rate = LEAGUE_AVG_HARD_HIT_FLY_RATE

        results[bid] = {
            "fly_ball_rate": round(fb_rate, 4),
            "hard_hit_fly_rate": round(hh_fly_rate, 4),
        }

    return results


def _compute_pitcher_recent_hr_vuln(
    conn: sqlite3.Connection,
    game_date: str,
    batter_ids: list[int],
    v1_pitcher_hr_vuln: dict[int, float],
) -> dict[int, float]:
    """Compute pitcher recent HR vulnerability for each batter's opposing pitcher.

    Uses the delta-cumulative approach over the pitcher's last N starts.

    Args:
        conn: Open sqlite3 connection.
        game_date: Evaluation date (YYYY-MM-DD).
        batter_ids: List of batter MLBAM IDs.
        v1_pitcher_hr_vuln: Dict of batter_id -> pitcher_hr_vuln from v1,
            used as fallback when the pitcher has < HR_PITCHER_RECENT_MIN_STARTS.

    Returns:
        Dict of batter_id -> pitcher_recent_hr_vuln (ratio to league avg).
    """
    if not batter_ids:
        return {}

    placeholders = ",".join("?" * len(batter_ids))

    # Find opposing pitchers for each batter on game_date
    opp_pitchers = conn.execute(f"""
        SELECT DISTINCT batter, pitcher
        FROM statcast_staging
        WHERE batter IN ({placeholders})
          AND game_date = ?
          AND events IS NOT NULL
    """, batter_ids + [game_date]).fetchall()

    batter_pitcher_map = defaultdict(set)
    for r in opp_pitchers:
        batter_pitcher_map[r["batter"]].add(r["pitcher"])

    # Fallback: if no data for prediction day, use most recent date
    if not batter_pitcher_map:
        max_date_row = conn.execute("""
            SELECT MAX(game_date) as md FROM statcast_staging WHERE game_date < ?
        """, (game_date,)).fetchone()
        recent_date = max_date_row["md"] if max_date_row else None
        if recent_date:
            opp_pitchers = conn.execute(f"""
                SELECT DISTINCT batter, pitcher
                FROM statcast_staging
                WHERE batter IN ({placeholders})
                  AND game_date = ?
                  AND events IS NOT NULL
            """, batter_ids + [recent_date]).fetchall()
            for r in opp_pitchers:
                batter_pitcher_map[r["batter"]].add(r["pitcher"])

    # Collect all unique pitcher IDs
    all_pitcher_ids = set()
    for pids in batter_pitcher_map.values():
        all_pitcher_ids.update(pids)

    # League average HR rate (for ratio computation)
    league_row = conn.execute("""
        SELECT SUM(hr_allowed) AS total_hr, SUM(batters_faced) AS total_bf
        FROM pitcher_stats
        WHERE game_date < ?
    """, (game_date,)).fetchone()
    league_hr_rate = 0.03
    if league_row and league_row["total_bf"] and league_row["total_bf"] > 0:
        league_hr_rate = (league_row["total_hr"] or 0) / league_row["total_bf"]

    # For each pitcher, compute recent HR vulnerability using delta-cumulative
    pitcher_recent_vuln = {}
    for pid in all_pitcher_ids:
        # Get the pitcher's most recent N game dates (starts) before evaluation
        recent_dates = conn.execute("""
            SELECT DISTINCT game_date
            FROM pitcher_stats
            WHERE pitcher_id = ? AND game_date < ?
            ORDER BY game_date DESC
            LIMIT ?
        """, (pid, game_date, HR_PITCHER_RECENT_STARTS)).fetchall()

        recent_game_dates = [r["game_date"] for r in recent_dates]

        if len(recent_game_dates) < HR_PITCHER_RECENT_MIN_STARTS:
            # Not enough recent starts -- will fall back to v1
            pitcher_recent_vuln[pid] = None
            continue

        # Delta-cumulative: stats between earliest and latest of the recent window
        earliest = min(recent_game_dates)
        latest = max(recent_game_dates)

        stats_row = conn.execute("""
            SELECT SUM(hr_allowed) AS total_hr, SUM(batters_faced) AS total_bf
            FROM pitcher_stats
            WHERE pitcher_id = ?
              AND game_date >= ?
              AND game_date <= ?
        """, (pid, earliest, latest)).fetchone()

        total_hr = (stats_row["total_hr"] or 0) if stats_row else 0
        total_bf = (stats_row["total_bf"] or 0) if stats_row else 0

        if total_bf > 0:
            recent_hr_rate = total_hr / total_bf
            ratio = recent_hr_rate / league_hr_rate if league_hr_rate > 0 else 1.0
            pitcher_recent_vuln[pid] = round(ratio, 3)
        else:
            pitcher_recent_vuln[pid] = None

    # Map back to batters
    results = {}
    for bid in batter_ids:
        opp_pids = batter_pitcher_map.get(bid, set())
        if opp_pids:
            vulns = [
                pitcher_recent_vuln[p]
                for p in opp_pids
                if p in pitcher_recent_vuln and pitcher_recent_vuln[p] is not None
            ]
            if vulns:
                results[bid] = round(float(np.mean(vulns)), 3)
            else:
                # Fall back to v1 pitcher_hr_vuln
                results[bid] = v1_pitcher_hr_vuln.get(bid, 1.0)
        else:
            results[bid] = v1_pitcher_hr_vuln.get(bid, 1.0)

    return results


def compute_hr_features_v2(
    conn: sqlite3.Connection,
    game_date: str,
    park_factors: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute HR v2 prediction features for all active batters on a date.

    Calls the v1 compute_hr_features() for the base 7 features, then extends
    with fly_ball_rate, hard_hit_fly_rate, and pitcher_recent_hr_vuln.

    Args:
        conn: Open sqlite3 connection.
        game_date: Date to compute features for (YYYY-MM-DD).
        park_factors: Pre-computed park factors. If None, loaded from table.

    Returns:
        DataFrame with columns: batter_id, barrel_rate, avg_exit_velo,
        hr_pa_rate, pitcher_hr_vuln, park_factor, platoon_hr_rate,
        power_trend, fly_ball_rate, hard_hit_fly_rate,
        pitcher_recent_hr_vuln, p_hr_v2.
    """
    # Step 1: Get base 7 features from v1
    v1_df = compute_hr_features(conn, game_date, park_factors=park_factors)
    if v1_df.empty:
        return pd.DataFrame()

    batter_ids = v1_df["batter_id"].tolist()

    # Step 2: Compute new features
    fb_features = _compute_fly_ball_features(conn, game_date, batter_ids)

    # Build v1 pitcher_hr_vuln lookup for fallback
    v1_pitcher_vuln = dict(
        zip(v1_df["batter_id"], v1_df["pitcher_hr_vuln"])
    )
    recent_vuln = _compute_pitcher_recent_hr_vuln(
        conn, game_date, batter_ids, v1_pitcher_vuln,
    )

    # Step 3: Merge new features into v1 DataFrame
    v1_df["fly_ball_rate"] = v1_df["batter_id"].map(
        lambda bid: fb_features.get(bid, {}).get(
            "fly_ball_rate", LEAGUE_AVG_FLY_BALL_RATE
        )
    )
    v1_df["hard_hit_fly_rate"] = v1_df["batter_id"].map(
        lambda bid: fb_features.get(bid, {}).get(
            "hard_hit_fly_rate", LEAGUE_AVG_HARD_HIT_FLY_RATE
        )
    )
    v1_df["pitcher_recent_hr_vuln"] = v1_df["batter_id"].map(
        lambda bid: recent_vuln.get(bid, 1.0)
    )

    # Step 4: Compute p_hr_v2 via expanded logistic model
    _fitted = load_hr_v2_coefficients()
    _intercept = _fitted["intercept"] if _fitted else HR_V2_INTERCEPT
    _weights = _fitted["weights"] if _fitted else HR_V2_WEIGHTS

    p_hr_v2_values = []
    for _, row in v1_df.iterrows():
        logit = _intercept
        logit += _weights["barrel_rate"] * (
            row["barrel_rate"] - _V2_FEATURE_BASELINES["barrel_rate"]
        )
        logit += _weights["avg_exit_velo"] * (
            row["avg_exit_velo"] - _V2_FEATURE_BASELINES["avg_exit_velo"]
        )
        logit += _weights["hr_pa_rate"] * (
            row["hr_pa_rate"] - _V2_FEATURE_BASELINES["hr_pa_rate"]
        )
        logit += _weights["pitcher_hr_vuln"] * (
            row["pitcher_hr_vuln"] - _V2_FEATURE_BASELINES["pitcher_hr_vuln"]
        )
        logit += _weights["park_factor"] * math.log(
            max(row["park_factor"], 0.5)
        )
        logit += _weights["platoon_hr_rate"] * (
            row["platoon_hr_rate"] - _V2_FEATURE_BASELINES["platoon_hr_rate"]
        )
        logit += _weights["power_trend"] * (
            row["power_trend"] - _V2_FEATURE_BASELINES["power_trend"]
        )
        logit += _weights["fly_ball_rate"] * (
            row["fly_ball_rate"] - _V2_FEATURE_BASELINES["fly_ball_rate"]
        )
        logit += _weights["hard_hit_fly_rate"] * (
            row["hard_hit_fly_rate"] - _V2_FEATURE_BASELINES["hard_hit_fly_rate"]
        )
        logit += _weights["pitcher_recent_hr_vuln"] * (
            row["pitcher_recent_hr_vuln"] - _V2_FEATURE_BASELINES["pitcher_recent_hr_vuln"]
        )

        # Convert logit to per-PA probability
        p_hr_per_pa = 1.0 / (1.0 + math.exp(-logit))

        # Scale to per-game probability (~3.5 PA per game)
        avg_pa_per_game = 3.5
        p_hr_game = 1.0 - (1.0 - p_hr_per_pa) ** avg_pa_per_game

        # Clamp to reasonable bounds
        p_hr_game = max(0.01, min(0.45, p_hr_game))

        p_hr_v2_values.append(round(p_hr_game, 4))

    v1_df["p_hr_v2"] = p_hr_v2_values

    # Drop the v1 p_hr column -- callers should use p_hr_v2
    v1_df = v1_df.drop(columns=["p_hr"], errors="ignore")

    if not v1_df.empty:
        logger.info(
            "HR v2 features for %s: %d batters, P(HR_v2) range [%.3f, %.3f], mean=%.3f",
            game_date, len(v1_df),
            v1_df["p_hr_v2"].min(), v1_df["p_hr_v2"].max(),
            v1_df["p_hr_v2"].mean(),
        )

    return v1_df


# ── Write Features ──────────────────────────────────────────────────────────

def write_hr_features_v2(
    conn: sqlite3.Connection,
    game_date: str,
    features_df: pd.DataFrame,
):
    """Write HR v2 features to the hr_features_v2 table.

    Args:
        conn: Open sqlite3 connection.
        game_date: Feature date.
        features_df: DataFrame from compute_hr_features_v2.
    """
    if features_df.empty:
        return

    conn.execute("DELETE FROM hr_features_v2 WHERE feature_date = ?", (game_date,))

    rows = []
    for _, row in features_df.iterrows():
        rows.append((
            game_date,
            int(row["batter_id"]),
            row["barrel_rate"],
            row["avg_exit_velo"],
            row["hr_pa_rate"],
            row["pitcher_hr_vuln"],
            row["park_factor"],
            row["platoon_hr_rate"],
            row["power_trend"],
            row["fly_ball_rate"],
            row["hard_hit_fly_rate"],
            row["pitcher_recent_hr_vuln"],
            row["p_hr_v2"],
        ))

    conn.executemany("""
        INSERT OR REPLACE INTO hr_features_v2
            (feature_date, batter_id, barrel_rate, avg_exit_velo, hr_pa_rate,
             pitcher_hr_vuln, park_factor, platoon_hr_rate, power_trend,
             fly_ball_rate, hard_hit_fly_rate, pitcher_recent_hr_vuln, p_hr_v2)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    logger.info("Wrote %d HR v2 feature rows for %s.", len(rows), game_date)


# ── CLI Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    print("=" * 60)
    print("HR Model V2 Coefficient Refit")
    print("=" * 60)

    print("\nDefault (hardcoded) v2 coefficients:")
    print(f"  intercept: {HR_V2_INTERCEPT}")
    for name in _V2_FEATURE_NAMES:
        print(f"  {name:25s}: {HR_V2_WEIGHTS[name]}")

    min_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    result = fit_hr_model_v2(conn, min_samples=min_samples)

    if result is None:
        print(f"\nInsufficient data (need >= {min_samples} rows). No refit performed.")
        sys.exit(1)

    print("\nFitted coefficients:")
    print(f"  intercept: {result['intercept']}")
    for name in _V2_FEATURE_NAMES:
        old = HR_V2_WEIGHTS[name]
        new = result["weights"][name]
        delta = new - old
        print(f"  {name:25s}: {new:+.6f}  (was {old:+.4f}, delta {delta:+.6f})")

    print(f"\nCoefficients saved to {HR_V2_COEFFICIENTS_PATH}")
    conn.close()
