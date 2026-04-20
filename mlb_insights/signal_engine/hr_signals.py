"""
mlb_insights/signal_engine/hr_signals.py -- HR-specific signal definitions.

Each signal function returns a SignalResult (or None if the signal does not
fire).  evaluate_hr_signals() runs all four HR signals for a single
batter-date.

Chunk 2 of the HR prediction pipeline.
"""

import logging
import sqlite3
from collections import defaultdict

from mlb_insights.signal_engine.signals import SignalResult
from mlb_insights.config import (
    HR_FB_POWER_MIN_FB_RATE,
    HR_FB_POWER_MIN_HHF_RATE,
    HR_FB_POWER_MIN_PARK,
    HR_PITCHER_FB_PERCENTILE,
    HR_POWER_STREAK_MIN_HR,
    HR_POWER_STREAK_LOOKBACK,
    HR_POWER_STREAK_BARREL_THRESHOLD,
    HR_PLATOON_MIN_HR_RATE,
)

logger = logging.getLogger(__name__)


# ── Helper Utilities ────────────────────────────────────────────────────────


def _clamp_confidence(val: float) -> float:
    """Clamp confidence to [0.1, 1.0] and round to 3 decimals."""
    return round(min(max(val, 0.1), 1.0), 3)


def compute_pitcher_fb_rates(
    conn: sqlite3.Connection, game_date: str
) -> dict[int, float]:
    """Precompute fly ball rate for all pitchers from statcast_staging.

    Fly ball rate is defined as the fraction of batted ball events (bb_type
    IS NOT NULL) where bb_type = 'fly_ball' for each pitcher, using only
    data strictly before *game_date*.

    Args:
        conn: Open sqlite3 connection.
        game_date: Date string (YYYY-MM-DD).  Only data before this date
            is considered.

    Returns:
        Dict mapping pitcher_id -> fly_ball_rate (float in [0, 1]).
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT pitcher,
               COUNT(*) as total_bbe,
               SUM(CASE WHEN bb_type = 'fly_ball' THEN 1 ELSE 0 END) as fly_balls
        FROM statcast_staging
        WHERE bb_type IS NOT NULL
          AND game_date < ?
        GROUP BY pitcher
        HAVING total_bbe >= 10
    """, (game_date,))

    rates = {}
    for r in cur.fetchall():
        total = r["total_bbe"]
        fb = r["fly_balls"]
        rates[r["pitcher"]] = fb / total if total > 0 else 0.0

    logger.debug(
        "compute_pitcher_fb_rates: computed rates for %d pitchers (before %s).",
        len(rates), game_date,
    )
    return rates


def get_recent_hr_count(
    conn: sqlite3.Connection,
    batter_id: int,
    game_date: str,
    lookback_games: int = 5,
) -> int:
    """Count HRs in last N games from batter_stats.

    Args:
        conn: Open sqlite3 connection.
        batter_id: Batter player ID.
        game_date: Date string (YYYY-MM-DD).  Only games strictly before
            this date are counted.
        lookback_games: Number of recent games to inspect.

    Returns:
        Total HR count in the lookback window.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT hr
        FROM batter_stats
        WHERE batter_id = ?
          AND game_date < ?
        ORDER BY game_date DESC
        LIMIT ?
    """, (batter_id, game_date, lookback_games))

    total_hr = 0
    for r in cur.fetchall():
        total_hr += (r["hr"] or 0)
    return total_hr


# ── Individual HR Signal Functions ──────────────────────────────────────────


def hr_fly_ball_power(
    hr_features: dict,
) -> SignalResult | None:
    """HR Signal 1: Fly-ball power in a hitter-friendly park.

    Fires when a batter combines a high fly ball rate, hard-hit fly ball
    rate, and a favorable park factor.

    Args:
        hr_features: Row from hr_features_v2 as dict.  Expected keys:
            fly_ball_rate, hard_hit_fly_rate, park_factor.

    Returns:
        SignalResult or None.
    """
    fly_ball_rate = hr_features.get("fly_ball_rate", 0) or 0
    hard_hit_fly_rate = hr_features.get("hard_hit_fly_rate", 0) or 0
    park_factor = hr_features.get("park_factor", 1.0) or 1.0

    if fly_ball_rate <= HR_FB_POWER_MIN_FB_RATE:
        return None
    if hard_hit_fly_rate <= HR_FB_POWER_MIN_HHF_RATE:
        return None
    if park_factor <= HR_FB_POWER_MIN_PARK:
        return None

    fb_factor = min((fly_ball_rate - HR_FB_POWER_MIN_FB_RATE) / 0.10, 1.0)
    hhf_factor = min((hard_hit_fly_rate - HR_FB_POWER_MIN_HHF_RATE) / 0.15, 1.0)
    park_factor_scaled = min((park_factor - HR_FB_POWER_MIN_PARK) / 0.20, 1.0)

    confidence = _clamp_confidence(
        0.4 * fb_factor + 0.3 * hhf_factor + 0.3 * park_factor_scaled
    )

    return SignalResult(
        signal_type="hr_fly_ball_power",
        confidence=confidence,
        headline=f"Fly-ball power profile in hitter-friendly park (PF {park_factor:.2f})",
        reasons=[
            f"Fly ball rate: {fly_ball_rate:.1%} (threshold: {HR_FB_POWER_MIN_FB_RATE:.0%})",
            f"Hard-hit fly rate: {hard_hit_fly_rate:.1%} (threshold: {HR_FB_POWER_MIN_HHF_RATE:.0%})",
            f"Park factor: {park_factor:.2f}",
        ],
        interpretation=(
            "Batter combines a high fly ball rate with hard-hit fly balls "
            "in a park that favors home runs"
        ),
    )


def hr_pitcher_flyball_tendency(
    pitcher_fb_rates: dict[int, float],
    opp_pitcher_ids: set[int],
) -> SignalResult | None:
    """HR Signal 2: Opposing pitcher has a high fly ball allowed rate.

    Fires when at least one opposing pitcher's fly ball rate exceeds the
    75th percentile among all pitchers in pitcher_fb_rates.

    Args:
        pitcher_fb_rates: Precomputed dict of pitcher_id -> fly_ball_rate.
        opp_pitcher_ids: Set of opposing pitcher IDs for the matchup.

    Returns:
        SignalResult or None.
    """
    if not pitcher_fb_rates or not opp_pitcher_ids:
        return None

    # Compute the 75th percentile threshold
    all_rates = sorted(pitcher_fb_rates.values())
    if not all_rates:
        return None
    idx_75 = int(len(all_rates) * HR_PITCHER_FB_PERCENTILE)
    idx_75 = min(idx_75, len(all_rates) - 1)
    percentile_75 = all_rates[idx_75]

    best_pitcher_id = None
    best_rate = 0.0
    for pid in opp_pitcher_ids:
        rate = pitcher_fb_rates.get(pid)
        if rate is not None and rate > percentile_75 and rate > best_rate:
            best_pitcher_id = pid
            best_rate = rate

    if best_pitcher_id is None:
        return None

    confidence = _clamp_confidence(
        min((best_rate - percentile_75) / 0.05, 1.0)
    )

    return SignalResult(
        signal_type="hr_pitcher_flyball_tendency",
        confidence=confidence,
        headline=f"Opposing pitcher allows high fly ball rate ({best_rate:.1%})",
        reasons=[
            f"Pitcher fly ball rate: {best_rate:.1%}",
            f"75th percentile threshold: {percentile_75:.1%}",
            f"Excess over threshold: +{best_rate - percentile_75:.1%}",
        ],
        interpretation=(
            "Opposing pitcher allows fly balls at an elevated rate, "
            "increasing home run opportunity"
        ),
    )


def hr_power_streak(
    conn: sqlite3.Connection,
    batter_id: int,
    game_date: str,
    hr_features: dict,
) -> SignalResult | None:
    """HR Signal 3: Recent power surge (HR count or barrel rate).

    Fires when a batter has hit 2+ HR in the last 5 games OR has a
    rolling barrel rate above 15% over recent batted ball events.

    Args:
        conn: Open sqlite3 connection.
        batter_id: Batter player ID.
        game_date: Date string (YYYY-MM-DD).
        hr_features: Row from hr_features_v2 as dict.  Expected keys:
            barrel_rate (optional).

    Returns:
        SignalResult or None.
    """
    recent_hr = get_recent_hr_count(
        conn, batter_id, game_date,
        lookback_games=HR_POWER_STREAK_LOOKBACK,
    )
    barrel_rate = hr_features.get("barrel_rate", 0) or 0

    hr_trigger = recent_hr >= HR_POWER_STREAK_MIN_HR
    barrel_trigger = barrel_rate > HR_POWER_STREAK_BARREL_THRESHOLD

    if not hr_trigger and not barrel_trigger:
        return None

    reasons = []
    if hr_trigger:
        # Confidence based on HR count: 2 = 0.5, 3+ = 0.8
        hr_confidence = 0.5 if recent_hr == 2 else 0.8
        reasons.append(
            f"{recent_hr} HR in last {HR_POWER_STREAK_LOOKBACK} games"
        )
    else:
        hr_confidence = 0.0

    if barrel_trigger:
        barrel_confidence = _clamp_confidence(
            min((barrel_rate - HR_POWER_STREAK_BARREL_THRESHOLD) / 0.10, 1.0)
        )
        reasons.append(
            f"Rolling barrel rate: {barrel_rate:.1%} "
            f"(threshold: {HR_POWER_STREAK_BARREL_THRESHOLD:.0%})"
        )
    else:
        barrel_confidence = 0.0

    # Take the higher confidence between the two triggers
    confidence = _clamp_confidence(max(hr_confidence, barrel_confidence))

    return SignalResult(
        signal_type="hr_power_streak",
        confidence=confidence,
        headline=f"Power surge: {recent_hr} HR in {HR_POWER_STREAK_LOOKBACK}G, barrel rate {barrel_rate:.1%}",
        reasons=reasons,
        interpretation=(
            "Batter is in a power streak with recent home runs or "
            "elevated barrel rate on batted balls"
        ),
    )


def hr_platoon_power(
    conn: sqlite3.Connection,
    batter_id: int,
    opp_pitcher_hand: str | None,
) -> SignalResult | None:
    """HR Signal 4: Platoon HR rate advantage against opposing hand.

    Fires when the batter's HR rate vs the opposing pitcher's handedness
    exceeds 5% and the matchup is favorable.

    Args:
        conn: Open sqlite3 connection.
        batter_id: Batter player ID.
        opp_pitcher_hand: 'L' or 'R' (opposing pitcher's throwing hand),
            or None if unknown.

    Returns:
        SignalResult or None.
    """
    if opp_pitcher_hand is None:
        return None

    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) as pa,
               SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) as hr
        FROM statcast_staging
        WHERE batter = ?
          AND p_throws = ?
          AND events IS NOT NULL
    """, (batter_id, opp_pitcher_hand))

    row = cur.fetchone()
    if not row or (row["pa"] or 0) == 0:
        return None

    pa = row["pa"]
    hr = row["hr"] or 0
    platoon_hr_rate = hr / pa

    if platoon_hr_rate <= HR_PLATOON_MIN_HR_RATE:
        return None

    confidence = _clamp_confidence(
        min((platoon_hr_rate - HR_PLATOON_MIN_HR_RATE) / 0.030, 1.0)
    )

    return SignalResult(
        signal_type="hr_platoon_power",
        confidence=confidence,
        headline=f"Platoon HR power vs {opp_pitcher_hand}HP ({platoon_hr_rate:.1%} HR rate)",
        reasons=[
            f"HR rate vs {opp_pitcher_hand}HP: {platoon_hr_rate:.1%} ({hr} HR in {pa} PA)",
            f"Minimum threshold: {HR_PLATOON_MIN_HR_RATE:.1%}",
            f"Facing {opp_pitcher_hand}-handed pitcher",
        ],
        interpretation=(
            f"Batter has an elevated home run rate against "
            f"{opp_pitcher_hand}-handed pitching and is facing the favorable matchup"
        ),
    )


# ── Evaluate All HR Signals ────────────────────────────────────────────────


def evaluate_hr_signals(
    conn: sqlite3.Connection,
    batter_id: int,
    game_date: str,
    hr_features: dict,
    pitcher_fb_rates: dict,
    opp_pitcher_ids: set,
    opp_pitcher_hand: str | None,
) -> list[SignalResult]:
    """Evaluate all HR-specific signals for a single batter on a given date.

    Args:
        conn: Open sqlite3 connection.
        batter_id: Batter player ID.
        game_date: Game date string (YYYY-MM-DD).
        hr_features: Row from hr_features_v2 as dict.
        pitcher_fb_rates: Precomputed dict of pitcher_id -> fly_ball_rate
            (from compute_pitcher_fb_rates).
        opp_pitcher_ids: Set of opposing pitcher IDs for the matchup.
        opp_pitcher_hand: 'L' or 'R' (opposing pitcher's throwing hand),
            or None if unknown.

    Returns:
        List of SignalResult objects for HR signals that fired.
    """
    fired = []

    # Signal 1: Fly-ball power
    try:
        result = hr_fly_ball_power(hr_features)
        if result is not None:
            fired.append(result)
    except Exception as exc:
        logger.warning(
            "Signal hr_fly_ball_power failed for batter %d on %s: %s",
            batter_id, game_date, exc,
        )

    # Signal 2: Pitcher fly-ball tendency
    try:
        result = hr_pitcher_flyball_tendency(pitcher_fb_rates, opp_pitcher_ids)
        if result is not None:
            fired.append(result)
    except Exception as exc:
        logger.warning(
            "Signal hr_pitcher_flyball_tendency failed for batter %d on %s: %s",
            batter_id, game_date, exc,
        )

    # Signal 3: Power streak
    try:
        result = hr_power_streak(conn, batter_id, game_date, hr_features)
        if result is not None:
            fired.append(result)
    except Exception as exc:
        logger.warning(
            "Signal hr_power_streak failed for batter %d on %s: %s",
            batter_id, game_date, exc,
        )

    # Signal 4: Platoon power
    try:
        result = hr_platoon_power(conn, batter_id, opp_pitcher_hand)
        if result is not None:
            fired.append(result)
    except Exception as exc:
        logger.warning(
            "Signal hr_platoon_power failed for batter %d on %s: %s",
            batter_id, game_date, exc,
        )

    return fired
