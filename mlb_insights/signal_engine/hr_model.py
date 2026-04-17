"""
mlb_insights/signal_engine/hr_model.py -- HR prediction sub-model.

Computes power-related features from statcast_staging data and produces
a calibrated P(HR) for each batter on a given date.

Features:
    1. Barrel rate (rolling 50 BBE)
    2. Average exit velocity (rolling 50 BBE)
    3. HR/PA rate (season)
    4. Pitcher HR vulnerability (opp pitcher hr_allowed/bf vs league)
    5. Park factor (HR rate per BBE by ballpark)
    6. Platoon HR rate (batter HR rate vs LHP/RHP)
    7. Power trend (recent exit velo vs long-term)

The final p_hr is computed via a logistic combination of features,
then scaled to reasonable probability bounds.
"""

import json
import logging
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_insights.config import DATA_DIR, DB_PATH

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BARREL_MIN_SPEED = 98.0
BARREL_BASE_ANGLE_MIN = 26.0
BARREL_BASE_ANGLE_MAX = 30.0
BARREL_ANGLE_MAX_CAP = 50.0
ROLLING_BBE_WINDOW = 50
ROLLING_BBE_SHORT = 10   # games for power trend (short window)

# League average reference points
LEAGUE_AVG_BARREL_RATE = 0.065
LEAGUE_AVG_EXIT_VELO = 88.0
LEAGUE_AVG_HR_PA_RATE = 0.035   # ~3.5% HR per PA
LEAGUE_AVG_HR_PER_GAME = 0.119  # ~11.9% chance of HR in a game

# Logistic model weights (tuned to produce ~3-4% base rate per PA,
# ~12% per game, with meaningful differentiation)
LOGISTIC_INTERCEPT = -3.25
LOGISTIC_WEIGHTS = {
    "barrel_rate":      3.5,
    "avg_exit_velo":    0.06,   # per mph above 88
    "hr_pa_rate":       8.0,
    "pitcher_hr_vuln":  2.5,
    "park_factor":      1.2,    # log(park_factor) scaled
    "platoon_hr_rate":  5.0,
    "power_trend":      1.5,
}

# Feature names in canonical order (must match LOGISTIC_WEIGHTS keys)
_FEATURE_NAMES = [
    "barrel_rate", "avg_exit_velo", "hr_pa_rate",
    "pitcher_hr_vuln", "park_factor", "platoon_hr_rate", "power_trend",
]

# Feature centering baselines (subtracted before model application)
_FEATURE_BASELINES = {
    "barrel_rate":     LEAGUE_AVG_BARREL_RATE,   # 0.065
    "avg_exit_velo":   LEAGUE_AVG_EXIT_VELO,     # 88.0
    "hr_pa_rate":      LEAGUE_AVG_HR_PA_RATE,    # 0.035
    "pitcher_hr_vuln": 1.0,
    "park_factor":     1.0,    # log(1.0) = 0
    "platoon_hr_rate": LEAGUE_AVG_HR_PA_RATE,    # 0.035
    "power_trend":     0.0,
}

HR_COEFFICIENTS_PATH = DATA_DIR / "hr_model_coefficients.json"


# ── Coefficient Persistence ─────────────────────────────────────────────────

def load_hr_coefficients() -> dict | None:
    """Load fitted HR model coefficients from disk.

    Returns:
        Dict with keys 'intercept' and 'weights' (dict of feature->coeff),
        or None if the file does not exist or is malformed.
    """
    if not HR_COEFFICIENTS_PATH.exists():
        return None
    try:
        with open(HR_COEFFICIENTS_PATH) as f:
            data = json.load(f)
        # Basic validation
        if "intercept" not in data or "weights" not in data:
            logger.warning("hr_model_coefficients.json missing required keys.")
            return None
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load HR coefficients: %s", exc)
        return None


def fit_hr_model(conn: sqlite3.Connection, min_samples: int = 500) -> dict | None:
    """Fit logistic regression on historical HR features vs actual outcomes.

    Pulls rows from hr_features joined with batter_stats to determine whether
    the batter actually hit a HR on that date. Centers features the same way
    the hardcoded model does, then fits sklearn LogisticRegression.

    Args:
        conn: Open sqlite3 connection.
        min_samples: Minimum rows required to fit.  Returns None if fewer.

    Returns:
        Dict with 'intercept' and 'weights', or None if insufficient data.
    """
    from sklearn.linear_model import LogisticRegression

    # Join hr_features with batter_stats to get actual HR outcome per game
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
            COALESCE(bs.home_runs, 0) AS actual_hr
        FROM hr_features hf
        LEFT JOIN batter_stats bs
            ON hf.batter_id = bs.batter_id
           AND hf.feature_date = bs.game_date
    """, conn)

    if len(df) < min_samples:
        logger.warning(
            "fit_hr_model: only %d rows (need %d). Skipping refit.",
            len(df), min_samples,
        )
        return None

    # Binary target: did batter hit at least 1 HR that game?
    y = (df["actual_hr"] >= 1).astype(int).values

    # Build centered feature matrix (same transforms as the hardcoded model)
    X = np.column_stack([
        df["barrel_rate"].values - _FEATURE_BASELINES["barrel_rate"],
        df["avg_exit_velo"].values - _FEATURE_BASELINES["avg_exit_velo"],
        df["hr_pa_rate"].values - _FEATURE_BASELINES["hr_pa_rate"],
        df["pitcher_hr_vuln"].values - _FEATURE_BASELINES["pitcher_hr_vuln"],
        np.log(np.clip(df["park_factor"].values, 0.5, None)),  # log transform
        df["platoon_hr_rate"].values - _FEATURE_BASELINES["platoon_hr_rate"],
        df["power_trend"].values - _FEATURE_BASELINES["power_trend"],
    ])

    logger.info(
        "fit_hr_model: fitting on %d samples (%d positive, %.2f%% HR rate).",
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
            for i, name in enumerate(_FEATURE_NAMES)
        },
    }

    # Save to disk
    HR_COEFFICIENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HR_COEFFICIENTS_PATH, "w") as f:
        json.dump(coefficients, f, indent=2)
    logger.info("Saved fitted HR coefficients to %s", HR_COEFFICIENTS_PATH)

    return coefficients


# ── Barrel Detection ────────────────────────────────────────────────────────

def is_barrel(launch_speed: float, launch_angle: float) -> bool:
    """Determine if a batted ball qualifies as a barrel.

    A barrel is launch_speed >= 98 AND launch_angle between 26-30 degrees,
    with the angle range expanding by 1 degree per mph above 98 (up to 50).

    Args:
        launch_speed: Exit velocity in mph.
        launch_angle: Launch angle in degrees.

    Returns:
        True if the batted ball is a barrel.
    """
    if launch_speed is None or launch_angle is None:
        return False
    if launch_speed < BARREL_MIN_SPEED:
        return False

    extra_degrees = launch_speed - BARREL_MIN_SPEED
    angle_min = BARREL_BASE_ANGLE_MIN
    angle_max = min(BARREL_BASE_ANGLE_MAX + extra_degrees, BARREL_ANGLE_MAX_CAP)

    return angle_min <= launch_angle <= angle_max


# ── Park Factors ─────────────────────────────────────────────────────────────

def compute_park_factors(conn: sqlite3.Connection) -> dict[str, float]:
    """Compute HR park factors from statcast_staging data.

    For each ballpark (identified by home_team), calculates the HR rate
    per batted ball event versus the league average. Uses all available
    data (typically 2+ seasons).

    Args:
        conn: Open sqlite3 connection.

    Returns:
        Dict of team -> park_factor (1.0 = average).
    """
    cur = conn.cursor()

    # Count HRs and total BBEs per ballpark
    cur.execute("""
        SELECT
            home_team,
            COUNT(*) AS total_bbe,
            SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hr_count
        FROM statcast_staging
        WHERE launch_speed IS NOT NULL
          AND launch_speed > 0
          AND home_team IS NOT NULL
          AND events IS NOT NULL
        GROUP BY home_team
        HAVING total_bbe >= 500
    """)

    park_data = {}
    total_hr = 0
    total_bbe = 0

    for row in cur.fetchall():
        team = row["home_team"]
        bbe = row["total_bbe"]
        hrs = row["hr_count"]
        park_data[team] = {"bbe": bbe, "hrs": hrs}
        total_hr += hrs
        total_bbe += bbe

    if total_bbe == 0:
        logger.warning("No batted ball data for park factors.")
        return {}

    league_hr_rate = total_hr / total_bbe

    park_factors = {}
    for team, data in park_data.items():
        team_hr_rate = data["hrs"] / data["bbe"] if data["bbe"] > 0 else 0
        factor = team_hr_rate / league_hr_rate if league_hr_rate > 0 else 1.0
        park_factors[team] = round(factor, 3)

    logger.info(
        "Computed park factors for %d teams. Range: [%.3f, %.3f]",
        len(park_factors),
        min(park_factors.values()) if park_factors else 0,
        max(park_factors.values()) if park_factors else 0,
    )

    return park_factors


def write_park_factors(conn: sqlite3.Connection, park_factors: dict[str, float]):
    """Write park factors to the park_factors table.

    Args:
        conn: Open sqlite3 connection.
        park_factors: Dict of team -> park_factor.
    """
    conn.execute("DELETE FROM park_factors")

    # Get sample sizes
    cur = conn.cursor()
    cur.execute("""
        SELECT home_team, COUNT(*) AS cnt
        FROM statcast_staging
        WHERE launch_speed IS NOT NULL AND launch_speed > 0
          AND events IS NOT NULL AND home_team IS NOT NULL
        GROUP BY home_team
    """)
    sample_sizes = {r["home_team"]: r["cnt"] for r in cur.fetchall()}

    rows = [
        (team, factor, sample_sizes.get(team, 0))
        for team, factor in park_factors.items()
    ]

    conn.executemany(
        "INSERT INTO park_factors (team, hr_park_factor, sample_size) VALUES (?,?,?)",
        rows,
    )
    conn.commit()
    logger.info("Wrote %d park factor rows.", len(rows))


# ── HR Feature Computation ───────────────────────────────────────────────────

def compute_hr_features(
    conn: sqlite3.Connection,
    game_date: str,
    park_factors: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute HR prediction features for all active batters on a date.

    Args:
        conn: Open sqlite3 connection.
        game_date: Date to compute features for (YYYY-MM-DD).
        park_factors: Pre-computed park factors. If None, loaded from table.

    Returns:
        DataFrame with columns: batter_id, barrel_rate, avg_exit_velo,
        hr_pa_rate, pitcher_hr_vuln, park_factor, platoon_hr_rate,
        power_trend, p_hr.
    """
    cur = conn.cursor()

    # Load fitted coefficients if available; fall back to hardcoded defaults
    _fitted = load_hr_coefficients()
    _intercept = _fitted["intercept"] if _fitted else LOGISTIC_INTERCEPT
    _weights = _fitted["weights"] if _fitted else LOGISTIC_WEIGHTS

    # Load park factors if not provided
    if park_factors is None:
        try:
            pf_rows = conn.execute("SELECT team, hr_park_factor FROM park_factors").fetchall()
            park_factors = {r["team"]: r["hr_park_factor"] for r in pf_rows}
        except Exception:
            park_factors = {}

    # Get active batters for this date (or most recent date if no data for exact date)
    batters = conn.execute("""
        SELECT DISTINCT batter_id, team FROM batter_stats WHERE game_date = ?
    """, (game_date,)).fetchall()

    if not batters:
        # Fall back to most recent date with data (for prediction-day lookups)
        batters = conn.execute("""
            SELECT DISTINCT batter_id, team FROM batter_stats
            WHERE game_date = (SELECT MAX(game_date) FROM batter_stats WHERE game_date <= ?)
        """, (game_date,)).fetchall()
        logger.info("HR model: no batters for %s, using %d from most recent date.",
                     game_date, len(batters))

    if not batters:
        return pd.DataFrame()

    batter_ids = [r["batter_id"] for r in batters]
    batter_teams = {r["batter_id"]: r["team"] for r in batters}

    # ── Load statcast BBE data for these batters (before game_date) ──────
    # For efficiency, load all at once
    placeholders = ",".join("?" * len(batter_ids))
    cur.execute(f"""
        SELECT game_date, batter, pitcher, events, launch_speed, launch_angle,
               p_throws, home_team, away_team, inning_topbot
        FROM statcast_staging
        WHERE batter IN ({placeholders})
          AND game_date < ?
          AND launch_speed IS NOT NULL
          AND launch_speed > 0
        ORDER BY batter, game_date, at_bat_number, pitch_number
    """, batter_ids + [game_date])

    # Organize by batter and derive team from most recent appearance
    batter_bbes = defaultdict(list)
    batter_recent_team = {}
    batter_recent_date = {}
    for row in cur.fetchall():
        d = dict(row)
        bid = row["batter"]
        batter_bbes[bid].append(d)
        rd = row["game_date"]
        if bid not in batter_recent_date or rd > batter_recent_date[bid]:
            batter_recent_date[bid] = rd
            # Derive team: if batting top inning -> away team, else home team
            if row["inning_topbot"] == "Top":
                batter_recent_team[bid] = row["away_team"]
            else:
                batter_recent_team[bid] = row["home_team"]

    # Fill batter_teams with derived teams where NULL
    for bid in batter_ids:
        if not batter_teams.get(bid) and bid in batter_recent_team:
            batter_teams[bid] = batter_recent_team[bid]

    # ── Load pitcher HR stats ────────────────────────────────────────────
    # Get opposing pitchers for this date (if statcast data exists for the date)
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

    # If no statcast for prediction day, use most recent game's opposing pitchers
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

    # Get all pitcher IDs we need
    all_pitcher_ids = set()
    for pids in batter_pitcher_map.values():
        all_pitcher_ids.update(pids)

    # Load pitcher HR vulnerability from pitcher_stats
    pitcher_hr_vuln = {}
    league_hr_bf = {"hr": 0, "bf": 0}

    if all_pitcher_ids:
        p_placeholders = ",".join("?" * len(all_pitcher_ids))
        pitcher_rows = conn.execute(f"""
            SELECT pitcher_id,
                   SUM(hr_allowed) AS total_hr,
                   SUM(batters_faced) AS total_bf
            FROM pitcher_stats
            WHERE pitcher_id IN ({p_placeholders})
              AND game_date < ?
            GROUP BY pitcher_id
            HAVING total_bf >= 50
        """, list(all_pitcher_ids) + [game_date]).fetchall()

        for r in pitcher_rows:
            bf = r["total_bf"] or 1
            hr = r["total_hr"] or 0
            pitcher_hr_vuln[r["pitcher_id"]] = hr / bf

    # League average pitcher HR rate
    league_row = conn.execute("""
        SELECT SUM(hr_allowed) AS total_hr, SUM(batters_faced) AS total_bf
        FROM pitcher_stats
        WHERE game_date < ?
    """, (game_date,)).fetchone()
    league_hr_rate_pitcher = 0.03
    if league_row and league_row["total_bf"] and league_row["total_bf"] > 0:
        league_hr_rate_pitcher = (league_row["total_hr"] or 0) / league_row["total_bf"]

    # ── Load season HR/PA for batters ────────────────────────────────────
    # Get season year from game_date
    season_year = game_date[:4]
    season_start = f"{season_year}-01-01"

    batter_season_hr = {}
    season_rows = conn.execute(f"""
        SELECT batter,
               SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hrs,
               COUNT(DISTINCT at_bat_number || '-' || game_pk) AS pa_approx
        FROM statcast_staging
        WHERE batter IN ({placeholders})
          AND game_date >= ?
          AND game_date < ?
          AND events IS NOT NULL
        GROUP BY batter
    """, batter_ids + [season_start, game_date]).fetchall()

    for r in season_rows:
        pa = r["pa_approx"] or 1
        batter_season_hr[r["batter"]] = (r["hrs"] or 0) / pa

    # ── Load platoon HR splits ───────────────────────────────────────────
    platoon_rows = conn.execute(f"""
        SELECT batter, p_throws,
               SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hrs,
               COUNT(*) AS pa
        FROM statcast_staging
        WHERE batter IN ({placeholders})
          AND game_date < ?
          AND events IS NOT NULL
          AND p_throws IS NOT NULL
        GROUP BY batter, p_throws
        HAVING pa >= 20
    """, batter_ids + [game_date]).fetchall()

    platoon_hr = defaultdict(dict)
    for r in platoon_rows:
        pa = r["pa"] or 1
        platoon_hr[r["batter"]][r["p_throws"]] = (r["hrs"] or 0) / pa

    # ── Determine opposing pitcher handedness ────────────────────────────
    # For each batter, find the primary pitcher they face on game_date
    batter_opp_hand = {}
    for bid, pids in batter_pitcher_map.items():
        for pid in pids:
            # Look up pitcher handedness
            hand_row = conn.execute("""
                SELECT p_throws FROM statcast_staging
                WHERE pitcher = ? AND p_throws IS NOT NULL
                LIMIT 1
            """, (pid,)).fetchone()
            if hand_row:
                batter_opp_hand[bid] = hand_row["p_throws"]
                break

    # ── Get game location for park factor ────────────────────────────────
    game_venues = {}
    venue_rows = conn.execute(f"""
        SELECT DISTINCT batter, home_team
        FROM statcast_staging
        WHERE batter IN ({placeholders})
          AND game_date = ?
    """, batter_ids + [game_date]).fetchall()

    for r in venue_rows:
        game_venues[r["batter"]] = r["home_team"]

    # If no venues found (future prediction date), derive from batter's team
    # Assume home game as rough approximation
    if not game_venues:
        for bid in batter_ids:
            team = batter_teams.get(bid)
            if team:
                game_venues[bid] = team

    # ── Compute features per batter ──────────────────────────────────────
    results = []

    for bid in batter_ids:
        bbes = batter_bbes.get(bid, [])

        # 1. Barrel rate (last 50 BBEs)
        recent_bbes = bbes[-ROLLING_BBE_WINDOW:]
        if len(recent_bbes) >= 10:
            barrels = sum(
                1 for b in recent_bbes
                if is_barrel(b.get("launch_speed"), b.get("launch_angle"))
            )
            barrel_rate = barrels / len(recent_bbes)
        else:
            barrel_rate = LEAGUE_AVG_BARREL_RATE

        # 2. Average exit velocity (last 50 BBEs)
        if len(recent_bbes) >= 10:
            speeds = [b["launch_speed"] for b in recent_bbes if b.get("launch_speed")]
            avg_exit_velo = np.mean(speeds) if speeds else LEAGUE_AVG_EXIT_VELO
        else:
            avg_exit_velo = LEAGUE_AVG_EXIT_VELO

        # 3. HR/PA rate (season)
        hr_pa_rate = batter_season_hr.get(bid, LEAGUE_AVG_HR_PA_RATE)

        # 4. Pitcher HR vulnerability
        opp_pids = batter_pitcher_map.get(bid, set())
        if opp_pids:
            vulns = [pitcher_hr_vuln[p] for p in opp_pids if p in pitcher_hr_vuln]
            if vulns:
                pitcher_vuln_val = np.mean(vulns) / league_hr_rate_pitcher if league_hr_rate_pitcher > 0 else 1.0
            else:
                pitcher_vuln_val = 1.0
        else:
            pitcher_vuln_val = 1.0

        # 5. Park factor
        venue = game_venues.get(bid)
        if venue and venue in park_factors:
            pf = park_factors[venue]
        else:
            # Try batter's team (home game assumption fallback)
            team = batter_teams.get(bid)
            pf = park_factors.get(team, 1.0) if team else 1.0

        # 6. Platoon HR rate
        opp_hand = batter_opp_hand.get(bid)
        if opp_hand and bid in platoon_hr and opp_hand in platoon_hr[bid]:
            plat_hr_rate = platoon_hr[bid][opp_hand]
        else:
            plat_hr_rate = hr_pa_rate  # fallback to overall rate

        # 7. Power trend (last 10 games exit velo vs last 50 games)
        if len(bbes) >= 20:
            # Group by game_date to get per-game averages
            game_speeds = defaultdict(list)
            for b in bbes:
                if b.get("launch_speed"):
                    game_speeds[b["game_date"]].append(b["launch_speed"])

            game_dates_sorted = sorted(game_speeds.keys())
            if len(game_dates_sorted) >= 10:
                recent_10_dates = game_dates_sorted[-10:]
                long_50_dates = game_dates_sorted[-50:] if len(game_dates_sorted) >= 50 else game_dates_sorted

                recent_speeds = [s for d in recent_10_dates for s in game_speeds[d]]
                long_speeds = [s for d in long_50_dates for s in game_speeds[d]]

                recent_avg = np.mean(recent_speeds) if recent_speeds else LEAGUE_AVG_EXIT_VELO
                long_avg = np.mean(long_speeds) if long_speeds else LEAGUE_AVG_EXIT_VELO

                power_trend = (recent_avg - long_avg) / long_avg if long_avg > 0 else 0.0
            else:
                power_trend = 0.0
        else:
            power_trend = 0.0

        # ── Compute P(HR) via logistic model ─────────────────────────────
        logit = _intercept
        logit += _weights["barrel_rate"] * (barrel_rate - LEAGUE_AVG_BARREL_RATE)
        logit += _weights["avg_exit_velo"] * (avg_exit_velo - LEAGUE_AVG_EXIT_VELO)
        logit += _weights["hr_pa_rate"] * (hr_pa_rate - LEAGUE_AVG_HR_PA_RATE)
        logit += _weights["pitcher_hr_vuln"] * (pitcher_vuln_val - 1.0)
        logit += _weights["park_factor"] * math.log(max(pf, 0.5))
        logit += _weights["platoon_hr_rate"] * (plat_hr_rate - LEAGUE_AVG_HR_PA_RATE)
        logit += _weights["power_trend"] * power_trend

        # Convert logit to per-PA probability
        p_hr_per_pa = 1.0 / (1.0 + math.exp(-logit))

        # Scale to per-game probability (~3.5 PA per game)
        # P(at least 1 HR in game) = 1 - (1 - p_hr_per_pa)^3.5
        avg_pa_per_game = 3.5
        p_hr_game = 1.0 - (1.0 - p_hr_per_pa) ** avg_pa_per_game

        # Clamp to reasonable bounds
        p_hr_game = max(0.01, min(0.45, p_hr_game))

        results.append({
            "batter_id": bid,
            "barrel_rate": round(barrel_rate, 4),
            "avg_exit_velo": round(float(avg_exit_velo), 1),
            "hr_pa_rate": round(hr_pa_rate, 4),
            "pitcher_hr_vuln": round(pitcher_vuln_val, 3),
            "park_factor": round(pf, 3),
            "platoon_hr_rate": round(plat_hr_rate, 4),
            "power_trend": round(power_trend, 4),
            "p_hr": round(p_hr_game, 4),
        })

    df = pd.DataFrame(results)
    if not df.empty:
        logger.info(
            "HR features for %s: %d batters, P(HR) range [%.3f, %.3f], mean=%.3f",
            game_date, len(df),
            df["p_hr"].min(), df["p_hr"].max(), df["p_hr"].mean(),
        )

    return df


def write_hr_features(
    conn: sqlite3.Connection,
    game_date: str,
    features_df: pd.DataFrame,
):
    """Write HR features to the hr_features table.

    Args:
        conn: Open sqlite3 connection.
        game_date: Feature date.
        features_df: DataFrame from compute_hr_features.
    """
    if features_df.empty:
        return

    conn.execute("DELETE FROM hr_features WHERE feature_date = ?", (game_date,))

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
            row["p_hr"],
        ))

    conn.executemany("""
        INSERT OR REPLACE INTO hr_features
            (feature_date, batter_id, barrel_rate, avg_exit_velo, hr_pa_rate,
             pitcher_hr_vuln, park_factor, platoon_hr_rate, power_trend, p_hr)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    logger.info("Wrote %d HR feature rows for %s.", len(rows), game_date)


# ── HR Power Signal ──────────────────────────────────────────────────────────

def check_hr_power_signal(
    barrel_rate: float,
    avg_exit_velo: float,
    pitcher_hr_vuln: float,
) -> bool:
    """Check if the HR power signal should fire.

    Fires when: barrel_rate > 10%, avg_exit_velo > 90, and pitcher
    has above-average HR vulnerability (> 1.0).

    Args:
        barrel_rate: Rolling barrel rate.
        avg_exit_velo: Rolling average exit velocity.
        pitcher_hr_vuln: Pitcher HR vulnerability ratio vs league.

    Returns:
        True if signal should fire.
    """
    return (
        barrel_rate > 0.10
        and avg_exit_velo > 90.0
        and pitcher_hr_vuln > 1.0
    )


# ── Barrel Rate History (for charts) ────────────────────────────────────────

def compute_barrel_rate_history(
    conn: sqlite3.Connection,
    batter_id: int,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Compute rolling barrel rate over time for a single batter.

    Returns a DataFrame with columns: game_date, barrel_rate, avg_exit_velo,
    suitable for charting in Streamlit.

    Args:
        conn: Open sqlite3 connection.
        batter_id: MLBAM batter ID.
        end_date: Optional end date filter.

    Returns:
        DataFrame with time series of barrel rate and exit velocity.
    """
    date_filter = f"AND game_date <= '{end_date}'" if end_date else ""

    cur = conn.cursor()
    cur.execute(f"""
        SELECT game_date, launch_speed, launch_angle
        FROM statcast_staging
        WHERE batter = ?
          AND launch_speed IS NOT NULL
          AND launch_speed > 0
          {date_filter}
        ORDER BY game_date, at_bat_number
    """, (batter_id,))

    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()

    # Group by game date
    game_bbes = defaultdict(list)
    for r in rows:
        game_bbes[r["game_date"]].append({
            "speed": r["launch_speed"],
            "angle": r["launch_angle"],
        })

    # Compute rolling metrics over time
    all_bbes = []
    results = []

    for date in sorted(game_bbes.keys()):
        for bbe in game_bbes[date]:
            all_bbes.append(bbe)

        # Rolling window
        window = all_bbes[-ROLLING_BBE_WINDOW:]
        if len(window) >= 10:
            barrels = sum(1 for b in window if is_barrel(b["speed"], b["angle"]))
            br = barrels / len(window)
            avg_velo = np.mean([b["speed"] for b in window])

            results.append({
                "game_date": date,
                "barrel_rate": round(br, 4),
                "avg_exit_velo": round(float(avg_velo), 1),
            })

    return pd.DataFrame(results)


# ── CLI Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    print("=" * 60)
    print("HR Model Coefficient Refit")
    print("=" * 60)

    print("\nDefault (hardcoded) coefficients:")
    print(f"  intercept: {LOGISTIC_INTERCEPT}")
    for name in _FEATURE_NAMES:
        print(f"  {name:20s}: {LOGISTIC_WEIGHTS[name]}")

    min_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    result = fit_hr_model(conn, min_samples=min_samples)

    if result is None:
        print(f"\nInsufficient data (need >= {min_samples} rows). No refit performed.")
        sys.exit(1)

    print("\nFitted coefficients:")
    print(f"  intercept: {result['intercept']}")
    for name in _FEATURE_NAMES:
        old = LOGISTIC_WEIGHTS[name]
        new = result["weights"][name]
        delta = new - old
        print(f"  {name:20s}: {new:+.6f}  (was {old:+.4f}, delta {delta:+.6f})")

    print(f"\nCoefficients saved to {HR_COEFFICIENTS_PATH}")
    conn.close()
