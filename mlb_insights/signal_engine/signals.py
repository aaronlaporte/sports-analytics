"""
mlb_insights/signal_engine/signals.py -- All 6 signal definitions.

Each signal function takes context dicts and returns a SignalResult (or None
if the signal does not fire).  evaluate_all_signals() runs them all for a
single batter-date.

Signals extracted from phase2b_rebuild.py Steps 4-5.
"""

import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from mlb_insights.config import (
    HOT_STREAK_MIN_STREAK, HOT_STREAK_MIN_HITS_PG5, HOT_STREAK_MIN_SEASON_AVG,
    HOT_STREAK_STREAK_CAP, HOT_STREAK_VOLUME_MIN, HOT_STREAK_VOLUME_SCALE,
    HOT_STREAK_SEASON_FLOOR, HOT_STREAK_SEASON_SCALE,
    COLD_STREAK_MAX_STREAK, COLD_STREAK_MIN_SEASON_AVG, COLD_STREAK_MIN_SEASON_PA,
    COLD_STREAK_MIN_EXIT_VELO, COLD_STREAK_VELO_SCALE, COLD_STREAK_GAP_SCALE,
    PITCHER_VULN_PERCENTILE, PITCHER_VULN_RATE_SCALE,
    CONTACT_QUALITY_MIN_SEASON_PA, CONTACT_QUALITY_MIN_SEASON_AVG,
    CONTACT_QUALITY_MIN_GAP, CONTACT_QUALITY_MIN_EXIT_VELO,
    CONTACT_QUALITY_GAP_SCALE, CONTACT_QUALITY_VELO_SCALE,
    PITCH_MIX_MIN_PA, PITCH_MIX_MIN_AVG, PITCH_MIX_MIN_ADVANTAGE,
    PITCH_MIX_SAMPLE_SCALE, PITCH_MIX_PERF_SCALE, PITCH_MIX_ADV_SCALE,
    BABIP_MIN_SEASON_PA, BABIP_MIN_GAP, BABIP_GAP_SCALE,
    BABIP_CONFIDENCE_SCALE, BABIP_APPROX_HR_RATE, BABIP_APPROX_SO_RATE,
    BABIP_APPROX_AB_RATIO,
    STREAK_CEILING_MIN_SEASON_PA, STREAK_CEILING_MIN_STREAK_INSTANCES,
    STREAK_CEILING_COLD_MIN_LEN, STREAK_CEILING_HOT_MIN_LEN,
    STREAK_CEILING_BUFFER,
    MIN_PITCHER_BF,
    LAUNCH_SPEED_N_GAMES, LAUNCH_SPEED_BATTED_BALLS_PER_GAME,
    LAUNCH_SPEED_MIN_SAMPLES,
)

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """A single signal that fired for a batter on a date."""
    signal_type: str
    confidence: float
    headline: str
    reasons: list[str]
    interpretation: str

    def to_db_tuple(self, signal_date: str, player_id: int) -> tuple:
        """Convert to a tuple for INSERT into daily_signals."""
        return (
            signal_date,
            player_id,
            self.signal_type,
            self.confidence,
            self.headline,
            json.dumps(self.reasons),
            self.interpretation,
        )


# ── Data Context Builder ─────────────────────────────────────────────────────

@dataclass
class SignalContext:
    """Pre-loaded data caches for signal evaluation on a given date."""
    batter_stats: dict          # (batter_id, date) -> row dict
    batter_dates: dict          # batter_id -> sorted list of dates
    pitcher_career: dict        # (pitcher_id, date) -> cumulative {bf, ha}
    league_avg_hit_rate: float
    batter_launch_speeds: dict  # batter_id -> [(date, speed), ...]
    batter_vs_hand: dict        # (batter_id, hand) -> {pa, hits, avg}
    pitcher_handedness: dict    # pitcher_id -> 'L' or 'R'
    batter_streak_history: dict = field(default_factory=dict)
    # batter_id -> {max_cold: int, max_hot: int,
    #               cold_counts: {len: count}, hot_counts: {len: count}}


def build_signal_context(conn: sqlite3.Connection) -> SignalContext:
    """Load all data caches needed for signal evaluation.

    This mirrors phase2b_rebuild.py Step 2 but returns a structured object
    instead of using module-level globals.

    Args:
        conn: Open sqlite3 connection.

    Returns:
        SignalContext with all caches populated.
    """
    cur = conn.cursor()

    # Batter stats cache
    logger.info("Loading batter_stats for signal context ...")
    cur.execute("""
        SELECT game_date, batter_id, pa, ab, hits, hr, bb, so,
               hits_pg_1, hits_pg_2, hits_pg_3, hits_pg_5, hits_pg_10, hits_pg_20,
               current_streak, season_hits, season_pa, season_hit_pct
        FROM batter_stats
        ORDER BY batter_id, game_date
    """)
    batter_stats = {}
    batter_dates = defaultdict(list)
    for r in cur.fetchall():
        key = (r["batter_id"], r["game_date"])
        batter_stats[key] = dict(r)
        batter_dates[r["batter_id"]].append(r["game_date"])

    # Pitcher career cumulative stats
    logger.info("Loading pitcher_stats for signal context ...")
    cur.execute("""
        SELECT pitcher_id, game_date, batters_faced, hits_allowed
        FROM pitcher_stats
        ORDER BY pitcher_id, game_date
    """)
    pitcher_rows = cur.fetchall()
    career = defaultdict(lambda: {"bf": 0, "ha": 0})
    pitcher_career = {}
    total_bf = 0
    total_ha = 0
    for r in pitcher_rows:
        pid = r["pitcher_id"]
        d = r["game_date"]
        bf = r["batters_faced"] or 0
        ha = r["hits_allowed"] or 0
        career[pid]["bf"] += bf
        career[pid]["ha"] += ha
        pitcher_career[(pid, d)] = dict(career[pid])
        total_bf += bf
        total_ha += ha

    league_avg = total_ha / total_bf if total_bf > 0 else 0.250

    # Launch speeds
    logger.info("Loading launch speed data ...")
    cur.execute("""
        SELECT game_date, batter, launch_speed
        FROM statcast_staging
        WHERE launch_speed IS NOT NULL
        ORDER BY batter, game_date
    """)
    batter_launch_speeds = defaultdict(list)
    for r in cur.fetchall():
        batter_launch_speeds[r["batter"]].append((r["game_date"], r["launch_speed"]))

    # Pitcher handedness
    logger.info("Loading pitcher handedness ...")
    cur.execute("""
        SELECT pitcher, p_throws, COUNT(*) as cnt
        FROM statcast_staging
        WHERE p_throws IS NOT NULL
        GROUP BY pitcher, p_throws
        ORDER BY pitcher, cnt DESC
    """)
    pitcher_handedness = {}
    for r in cur.fetchall():
        if r["pitcher"] not in pitcher_handedness:
            pitcher_handedness[r["pitcher"]] = r["p_throws"]

    # Batter vs handedness splits
    logger.info("Computing batter vs handedness splits ...")
    cur.execute("""
        SELECT batter, p_throws,
               COUNT(*) as pa,
               SUM(CASE WHEN events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END) as hits
        FROM statcast_staging
        WHERE events IS NOT NULL
        GROUP BY batter, p_throws
    """)
    batter_vs_hand = {}
    for r in cur.fetchall():
        pa = r["pa"]
        hits = r["hits"]
        batter_vs_hand[(r["batter"], r["p_throws"])] = {
            "pa": pa, "hits": hits, "avg": hits / pa if pa > 0 else 0.0,
        }

    # Streak history: compute max cold/hot streaks and frequency per batter
    logger.info("Computing streak ceiling history ...")
    batter_streak_history = _compute_streak_history(batter_stats, batter_dates)

    logger.info(
        "Signal context ready: %d batter-dates, %d pitcher-dates, %d launch batters, %d streak histories.",
        len(batter_stats), len(pitcher_career), len(batter_launch_speeds),
        len(batter_streak_history),
    )

    return SignalContext(
        batter_stats=batter_stats,
        batter_dates=batter_dates,
        pitcher_career=pitcher_career,
        league_avg_hit_rate=league_avg,
        batter_launch_speeds=batter_launch_speeds,
        batter_vs_hand=batter_vs_hand,
        pitcher_handedness=pitcher_handedness,
        batter_streak_history=batter_streak_history,
    )


# ── Context Lookup Helpers ───────────────────────────────────────────────────

def _get_batter_stats(ctx: SignalContext, batter_id: int, date: str) -> dict | None:
    """Get most recent batter_stats on or before date."""
    key = (batter_id, date)
    if key in ctx.batter_stats:
        return ctx.batter_stats[key]
    dates = ctx.batter_dates.get(batter_id, [])
    prev = None
    for d in dates:
        if d <= date:
            prev = d
        else:
            break
    if prev:
        return ctx.batter_stats.get((batter_id, prev))
    return None


def _get_avg_launch_speed(ctx: SignalContext, batter_id: int, date: str) -> float | None:
    """Get average launch speed for batter over recent games before date."""
    speeds = ctx.batter_launch_speeds.get(batter_id, [])
    if not speeds:
        return None
    recent = [s for d, s in speeds if d < date]
    if len(recent) < LAUNCH_SPEED_MIN_SAMPLES:
        return None
    recent = recent[-(LAUNCH_SPEED_N_GAMES * LAUNCH_SPEED_BATTED_BALLS_PER_GAME):]
    return float(np.mean(recent)) if recent else None


def _get_pitcher_vulnerability(ctx: SignalContext, date: str) -> dict:
    """Compute pitcher vulnerability rankings for a date.

    Returns:
        Dict of pitcher_id -> (avg_against, percentile).
    """
    pitcher_avgs = {}
    for (pid, d), stats in ctx.pitcher_career.items():
        if d <= date and stats["bf"] >= MIN_PITCHER_BF:
            pitcher_avgs[pid] = stats["ha"] / stats["bf"]

    if not pitcher_avgs:
        return {}

    values = sorted(pitcher_avgs.values())
    n = len(values)
    result = {}
    for pid, avg in pitcher_avgs.items():
        rank = sum(1 for v in values if v <= avg) / n
        result[pid] = (avg, rank)
    return result


def _compute_recent_babip(ctx: SignalContext, batter_id: int, date: str):
    """Compute approximate recent and season BABIP.

    Returns:
        (recent_babip, season_babip) or (None, None) if insufficient data.
    """
    bs = _get_batter_stats(ctx, batter_id, date)
    if not bs:
        return None, None

    recent_hits = bs.get("hits_pg_5", 0) or 0
    season_pa = bs.get("season_pa", 0) or 0
    season_hits = bs.get("season_hits", 0) or 0
    season_hit_pct = bs.get("season_hit_pct", 0) or 0

    if season_pa < BABIP_MIN_SEASON_PA:
        return None, None

    # Recent BABIP approximation (5 games ~ 20 AB)
    approx_recent_ab = 20
    approx_recent_hr = recent_hits * BABIP_APPROX_HR_RATE
    approx_recent_so = approx_recent_ab * BABIP_APPROX_SO_RATE

    denom = approx_recent_ab - approx_recent_so - approx_recent_hr
    if denom <= 0:
        return None, None
    recent_babip = (recent_hits - approx_recent_hr) / denom

    # Season BABIP approximation
    season_babip = None
    if season_pa > 0:
        season_ab = season_pa * BABIP_APPROX_AB_RATIO
        season_hr = season_hits * BABIP_APPROX_HR_RATE
        season_so = season_ab * BABIP_APPROX_SO_RATE
        s_denom = season_ab - season_so - season_hr
        if s_denom > 0:
            season_babip = (season_hits - season_hr) / s_denom

    return recent_babip, season_babip


# ── Individual Signal Functions ──────────────────────────────────────────────

def _clamp_confidence(val: float) -> float:
    return round(min(max(val, 0.1), 1.0), 3)


def hot_streak_acceleration(
    ctx: SignalContext, batter_id: int, date: str
) -> SignalResult | None:
    """Signal 1: Batter in a sustained hot streak with high-volume contact."""
    bs = _get_batter_stats(ctx, batter_id, date)
    if not bs:
        return None

    streak = bs.get("current_streak", 0) or 0
    hits_pg_5 = bs.get("hits_pg_5", 0) or 0
    season_hit_pct = bs.get("season_hit_pct", 0) or 0

    if streak < HOT_STREAK_MIN_STREAK:
        return None
    if hits_pg_5 < HOT_STREAK_MIN_HITS_PG5:
        return None
    if season_hit_pct <= HOT_STREAK_MIN_SEASON_AVG:
        return None

    streak_factor = min(streak / HOT_STREAK_STREAK_CAP, 1.0)
    volume_factor = min((hits_pg_5 - HOT_STREAK_VOLUME_MIN) / HOT_STREAK_VOLUME_SCALE, 1.0)
    season_factor = min(
        (season_hit_pct - HOT_STREAK_SEASON_FLOOR) / HOT_STREAK_SEASON_SCALE, 1.0
    )
    confidence = _clamp_confidence(
        0.4 * streak_factor + 0.35 * volume_factor + 0.25 * season_factor
    )

    return SignalResult(
        signal_type="hot_streak_acceleration",
        confidence=confidence,
        headline=f"Hot streak ({streak}G) with high-volume contact",
        reasons=[
            f"Hit streak: {streak} games",
            f"Recent volume: {hits_pg_5} hits in 5 games",
            f"Season avg: {season_hit_pct:.3f}",
        ],
        interpretation="Batter is in a sustained hot streak with genuine quality at-bats",
    )


def cold_streak_rebound(
    ctx: SignalContext, batter_id: int, date: str
) -> SignalResult | None:
    """Signal 2: Quality hitter in cold streak -- no launch speed requirement.

    Note: phase2b used launch speed as a gate. Per spec, this signal now fires
    without the launch speed requirement, but we still use it for confidence
    weighting when available.
    """
    bs = _get_batter_stats(ctx, batter_id, date)
    if not bs:
        return None

    streak = bs.get("current_streak", 0) or 0
    season_hit_pct = bs.get("season_hit_pct", 0) or 0
    season_pa = bs.get("season_pa", 0) or 0

    if streak > COLD_STREAK_MAX_STREAK:  # streak is negative, so > -5 means not cold enough
        return None
    if season_hit_pct < COLD_STREAK_MIN_SEASON_AVG:
        return None
    if season_pa < COLD_STREAK_MIN_SEASON_PA:
        return None

    hits_pg_5 = bs.get("hits_pg_5", 0) or 0
    gap = season_hit_pct - (hits_pg_5 / 20.0 if hits_pg_5 else 0)
    gap_factor = min(gap / COLD_STREAK_GAP_SCALE, 1.0) if gap > 0 else 0.0

    # Use launch speed for confidence boost when available, but don't gate on it
    avg_ls = _get_avg_launch_speed(ctx, batter_id, date)
    if avg_ls is not None and avg_ls > COLD_STREAK_MIN_EXIT_VELO:
        ls_factor = min((avg_ls - COLD_STREAK_MIN_EXIT_VELO) / COLD_STREAK_VELO_SCALE, 1.0)
        confidence = _clamp_confidence(0.5 * ls_factor + 0.5 * gap_factor)
        reasons = [
            f"Cold streak: {abs(streak)} games without a hit",
            f"Season avg: {season_hit_pct:.3f} (quality hitter)",
            f"Avg exit velocity (10G): {avg_ls:.1f} mph (hard contact)",
        ]
    else:
        # No launch speed data -- use gap and season quality only
        season_factor = min(
            (season_hit_pct - COLD_STREAK_MIN_SEASON_AVG) / 0.040, 1.0
        )
        confidence = _clamp_confidence(0.4 * gap_factor + 0.6 * season_factor)
        reasons = [
            f"Cold streak: {abs(streak)} games without a hit",
            f"Season avg: {season_hit_pct:.3f} (quality hitter)",
        ]

    return SignalResult(
        signal_type="cold_streak_rebound",
        confidence=confidence,
        headline="Quality hitter in cold streak with strong contact metrics",
        reasons=reasons,
        interpretation="Good hitter making hard contact but getting unlucky - rebound expected",
    )


def pitcher_vulnerability(
    ctx: SignalContext,
    batter_id: int,
    date: str,
    pitchers_faced: set[int] | None = None,
    pitcher_vuln: dict | None = None,
) -> SignalResult | None:
    """Signal 3: Opposing pitcher allows hits at above-average rate.

    Args:
        ctx: Signal context.
        batter_id: Batter player ID.
        date: Game date string.
        pitchers_faced: Set of pitcher IDs this batter faced on this date.
        pitcher_vuln: Pre-computed pitcher vulnerability dict for this date.
    """
    if pitcher_vuln is None:
        pitcher_vuln = _get_pitcher_vulnerability(ctx, date)
    if not pitcher_vuln:
        return None

    if pitchers_faced is None:
        return None

    for pid in pitchers_faced:
        if pid not in pitcher_vuln:
            continue
        avg_against, percentile = pitcher_vuln[pid]
        if percentile >= PITCHER_VULN_PERCENTILE:
            vuln_factor = min((percentile - PITCHER_VULN_PERCENTILE) / 0.25, 1.0)
            rate_factor = (
                min((avg_against - ctx.league_avg_hit_rate) / PITCHER_VULN_RATE_SCALE, 1.0)
                if avg_against > ctx.league_avg_hit_rate
                else 0.0
            )
            confidence = _clamp_confidence(0.5 * vuln_factor + 0.5 * rate_factor)

            return SignalResult(
                signal_type="pitcher_vulnerability",
                confidence=confidence,
                headline=f"Facing vulnerable pitcher (top {(1 - percentile) * 100:.0f}% hittable)",
                reasons=[
                    f"Opposing pitcher avg against: {avg_against:.3f}",
                    f"Pitcher vulnerability percentile: {percentile:.0%} (top {(1 - percentile) * 100:.0f}%)",
                    f"League avg hit rate: {ctx.league_avg_hit_rate:.3f}",
                ],
                interpretation="Opposing pitcher allows hits at above-average rate",
            )

    return None


def contact_quality_regression(
    ctx: SignalContext, batter_id: int, date: str
) -> SignalResult | None:
    """Signal 4: Hard contact but low hit rate -- regression to mean expected."""
    bs = _get_batter_stats(ctx, batter_id, date)
    if not bs:
        return None

    season_pa = bs.get("season_pa", 0) or 0
    season_hit_pct = bs.get("season_hit_pct", 0) or 0
    hits_pg_5 = bs.get("hits_pg_5", 0) or 0

    if season_pa < CONTACT_QUALITY_MIN_SEASON_PA:
        return None
    if season_hit_pct <= CONTACT_QUALITY_MIN_SEASON_AVG:
        return None

    recent_hit_rate = hits_pg_5 / 20.0 if hits_pg_5 else 0
    gap = season_hit_pct - recent_hit_rate
    if gap <= CONTACT_QUALITY_MIN_GAP:
        return None

    avg_ls = _get_avg_launch_speed(ctx, batter_id, date)
    if avg_ls is None or avg_ls <= CONTACT_QUALITY_MIN_EXIT_VELO:
        return None

    gap_factor = min(gap / CONTACT_QUALITY_GAP_SCALE, 1.0)
    ls_factor = min((avg_ls - CONTACT_QUALITY_MIN_EXIT_VELO) / CONTACT_QUALITY_VELO_SCALE, 1.0)
    confidence = _clamp_confidence(0.5 * gap_factor + 0.5 * ls_factor)

    return SignalResult(
        signal_type="contact_quality_regression",
        confidence=confidence,
        headline="Unlucky stretch - hard contact but low hit rate",
        reasons=[
            f"Recent hit rate ({recent_hit_rate:.3f}) well below season ({season_hit_pct:.3f})",
            f"Avg exit velocity: {avg_ls:.1f} mph (quality contact)",
            f"Performance gap: {gap:.3f}",
        ],
        interpretation="Batter making solid contact but results not reflecting quality - regression to mean expected",
    )


def pitch_mix_advantage(
    ctx: SignalContext, batter_id: int, date: str
) -> SignalResult | None:
    """Signal 5: Batter has proven advantage against a pitcher handedness type."""
    bs = _get_batter_stats(ctx, batter_id, date)
    if not bs:
        return None

    season_hit_pct = bs.get("season_hit_pct", 0) or 0

    for hand in ["L", "R"]:
        split_key = (batter_id, hand)
        if split_key not in ctx.batter_vs_hand:
            continue
        split = ctx.batter_vs_hand[split_key]
        if split["pa"] < PITCH_MIX_MIN_PA:
            continue
        if split["avg"] < PITCH_MIX_MIN_AVG:
            continue

        advantage = split["avg"] - season_hit_pct
        if advantage <= PITCH_MIX_MIN_ADVANTAGE:
            continue

        sample_factor = min(split["pa"] / PITCH_MIX_SAMPLE_SCALE, 1.0)
        perf_factor = min((split["avg"] - PITCH_MIX_MIN_AVG) / PITCH_MIX_PERF_SCALE, 1.0)
        adv_factor = min(advantage / PITCH_MIX_ADV_SCALE, 1.0)
        confidence = _clamp_confidence(
            0.3 * sample_factor + 0.4 * perf_factor + 0.3 * adv_factor
        )

        return SignalResult(
            signal_type="pitch_mix_advantage",
            confidence=confidence,
            headline=f"Strong platoon advantage vs {hand}HP",
            reasons=[
                f"Batting {split['avg']:.3f} vs {hand}HP ({split['pa']} PA)",
                f"Overall avg: {season_hit_pct:.3f}",
                f"Split advantage: +{advantage:.3f}",
            ],
            interpretation=f"Batter has proven advantage against {hand}-handed pitching",
        )

    return None


def babip_regression(
    ctx: SignalContext, batter_id: int, date: str
) -> SignalResult | None:
    """Signal 6: Recent BABIP significantly below season BABIP."""
    bs = _get_batter_stats(ctx, batter_id, date)
    if not bs:
        return None

    season_pa = bs.get("season_pa", 0) or 0
    if season_pa < BABIP_MIN_SEASON_PA:
        return None

    recent_babip, season_babip = _compute_recent_babip(ctx, batter_id, date)
    if recent_babip is None or season_babip is None:
        return None

    babip_gap = season_babip - recent_babip
    if babip_gap <= BABIP_MIN_GAP:
        return None

    gap_factor = min(babip_gap / BABIP_GAP_SCALE, 1.0)
    confidence = _clamp_confidence(gap_factor * BABIP_CONFIDENCE_SCALE)

    return SignalResult(
        signal_type="babip_regression",
        confidence=confidence,
        headline=f"BABIP regression candidate (gap: {babip_gap:.3f})",
        reasons=[
            f"Recent BABIP: {recent_babip:.3f}",
            f"Season BABIP: {season_babip:.3f}",
            f"BABIP gap: {babip_gap:.3f} (regression candidate)",
        ],
        interpretation="Recent BABIP significantly below season norm - positive regression expected",
    )


# ── Streak Ceiling Signals ──────────────────────────────────────────────────


def _compute_streak_history(
    batter_stats: dict, batter_dates: dict
) -> dict:
    """Compute historical max streaks and streak-length frequency per batter.

    Walks through each batter's game-by-game streak values and tracks:
    - How many times they've had a cold streak of length N
    - How many times they've had a hot streak of length N
    - Their all-time max cold and hot streak lengths

    This mirrors the user's derive_streak_summary_scaled() logic.

    Returns:
        Dict of batter_id -> {
            max_cold: int,  max_hot: int,
            cold_counts: {length: count}, hot_counts: {length: count}
        }
    """
    result = {}

    for batter_id, dates in batter_dates.items():
        max_cold = 0
        max_hot = 0
        cold_counts = defaultdict(int)  # length -> how many times reached
        hot_counts = defaultdict(int)

        for d in dates:
            bs = batter_stats.get((batter_id, d))
            if not bs:
                continue
            streak = bs.get("current_streak")
            if streak is None or not isinstance(streak, (int, float)):
                continue
            streak = int(streak)

            if streak < 0:
                cold_len = abs(streak)
                if cold_len > max_cold:
                    max_cold = cold_len
                # Count: this date was part of a cold streak of at least cold_len
                for length in range(STREAK_CEILING_COLD_MIN_LEN, cold_len + 1):
                    cold_counts[length] += 1
            elif streak > 0:
                if streak > max_hot:
                    max_hot = streak
                for length in range(STREAK_CEILING_HOT_MIN_LEN, streak + 1):
                    hot_counts[length] += 1

        result[batter_id] = {
            "max_cold": max_cold,
            "max_hot": max_hot,
            "cold_counts": dict(cold_counts),
            "hot_counts": dict(hot_counts),
        }

    return result


def streak_ceiling_cold(
    ctx: SignalContext, batter_id: int, date: str
) -> SignalResult | None:
    """Signal 7: Batter's cold streak is at or near their historical max.

    If a player has never gone more than 7 games without a hit, and they're
    currently at game 7, history says they'll rebound. The signal fires when
    the current cold streak is within STREAK_CEILING_BUFFER of their all-time
    max cold streak.

    Higher confidence when:
    - They've hit this ceiling multiple times before (pattern is reliable)
    - They're exactly at the max (not just approaching)
    - They're a quality hitter (higher season avg)
    """
    bs = _get_batter_stats(ctx, batter_id, date)
    if not bs:
        return None

    streak = bs.get("current_streak", 0) or 0
    if streak >= 0:
        return None  # Not in a cold streak

    cold_len = abs(streak)
    if cold_len < STREAK_CEILING_COLD_MIN_LEN:
        return None

    season_pa = bs.get("season_pa", 0) or 0
    if season_pa < STREAK_CEILING_MIN_SEASON_PA:
        return None

    history = ctx.batter_streak_history.get(batter_id)
    if not history:
        return None

    max_cold = history["max_cold"]
    if max_cold < STREAK_CEILING_COLD_MIN_LEN:
        return None  # Not enough streak history

    # Fire if current cold streak is within BUFFER of historical max
    if cold_len < max_cold - STREAK_CEILING_BUFFER:
        return None  # Not close enough to ceiling

    # Check that they've hit this streak length enough times for pattern reliability
    times_at_this_length = history["cold_counts"].get(cold_len, 0)
    if times_at_this_length < STREAK_CEILING_MIN_STREAK_INSTANCES:
        return None  # Not enough historical instances

    # Confidence factors
    # 1. Proximity to ceiling (at max = 1.0, one game away = 0.6)
    proximity = 1.0 if cold_len >= max_cold else 0.6
    # 2. Pattern reliability (more instances = more reliable)
    pattern_factor = min(times_at_this_length / 8.0, 1.0)
    # 3. Hitter quality (better hitters rebound more reliably)
    season_hit_pct = bs.get("season_hit_pct", 0) or 0
    quality_factor = min(max(season_hit_pct - 0.200, 0) / 0.100, 1.0)

    confidence = _clamp_confidence(
        0.40 * proximity + 0.35 * pattern_factor + 0.25 * quality_factor
    )

    at_or_near = "at" if cold_len >= max_cold else "near"

    return SignalResult(
        signal_type="streak_ceiling_cold",
        confidence=confidence,
        headline=f"Cold streak {at_or_near} historical ceiling ({cold_len}G of max {max_cold}G)",
        reasons=[
            f"Current hitless streak: {cold_len} games",
            f"Historical max cold streak: {max_cold} games",
            f"Times reached {cold_len}+ game cold streak: {times_at_this_length}x (never exceeded {max_cold})",
            f"Season avg: {season_hit_pct:.3f}",
        ],
        interpretation=(
            f"Player has hit a {cold_len}-game cold streak {times_at_this_length} times "
            f"and never gone past {max_cold} — historical pattern says rebound is due"
        ),
    )


def streak_ceiling_hot(
    ctx: SignalContext, batter_id: int, date: str
) -> SignalResult | None:
    """Signal 8: Batter's hot streak is at or near their historical max.

    If a player's longest hit streak is 10 games and they're at game 10,
    regression says the streak is likely to end. This is a negative signal
    (slightly lowers expected hit probability).

    Note: This signal has NEGATIVE weight in the composite score — it reduces
    the daily score rather than boosting it.
    """
    bs = _get_batter_stats(ctx, batter_id, date)
    if not bs:
        return None

    streak = bs.get("current_streak", 0) or 0
    if streak <= 0:
        return None  # Not in a hot streak

    if streak < STREAK_CEILING_HOT_MIN_LEN:
        return None

    season_pa = bs.get("season_pa", 0) or 0
    if season_pa < STREAK_CEILING_MIN_SEASON_PA:
        return None

    history = ctx.batter_streak_history.get(batter_id)
    if not history:
        return None

    max_hot = history["max_hot"]
    if max_hot < STREAK_CEILING_HOT_MIN_LEN:
        return None

    # Fire if current hot streak is within BUFFER of historical max
    if streak < max_hot - STREAK_CEILING_BUFFER:
        return None

    times_at_this_length = history["hot_counts"].get(streak, 0)
    if times_at_this_length < STREAK_CEILING_MIN_STREAK_INSTANCES:
        return None

    # Confidence factors
    proximity = 1.0 if streak >= max_hot else 0.6
    pattern_factor = min(times_at_this_length / 6.0, 1.0)

    confidence = _clamp_confidence(
        0.50 * proximity + 0.50 * pattern_factor
    )

    at_or_near = "at" if streak >= max_hot else "near"

    return SignalResult(
        signal_type="streak_ceiling_hot",
        confidence=confidence,
        headline=f"Hot streak {at_or_near} historical ceiling ({streak}G of max {max_hot}G)",
        reasons=[
            f"Current hit streak: {streak} games",
            f"Historical max hot streak: {max_hot} games",
            f"Times reached {streak}+ game hit streak: {times_at_this_length}x",
        ],
        interpretation=(
            f"Player's hit streak ({streak}G) is {at_or_near} their historical max "
            f"of {max_hot}G — regression says the streak is likely to end"
        ),
    )


# ── Evaluate All Signals ─────────────────────────────────────────────────────

ALL_SIGNAL_FUNCTIONS = [
    hot_streak_acceleration,
    cold_streak_rebound,
    # pitcher_vulnerability handled separately (needs pitchers_faced)
    contact_quality_regression,
    pitch_mix_advantage,
    babip_regression,
    streak_ceiling_cold,
    streak_ceiling_hot,
]


def evaluate_all_signals(
    ctx: SignalContext,
    batter_id: int,
    date: str,
    pitchers_faced: set[int] | None = None,
    pitcher_vuln: dict | None = None,
) -> list[SignalResult]:
    """Evaluate all 6 signals for a single batter on a given date.

    Args:
        ctx: Pre-built SignalContext.
        batter_id: Batter player ID.
        date: Game date string (YYYY-MM-DD).
        pitchers_faced: Set of pitcher IDs the batter faced on this date.
        pitcher_vuln: Pre-computed pitcher vulnerability dict for this date.

    Returns:
        List of SignalResult objects for signals that fired.
    """
    fired = []

    for fn in ALL_SIGNAL_FUNCTIONS:
        try:
            result = fn(ctx, batter_id, date)
            if result is not None:
                fired.append(result)
        except Exception as exc:
            logger.warning(
                "Signal %s failed for batter %d on %s: %s",
                fn.__name__, batter_id, date, exc,
            )

    # Pitcher vulnerability (needs extra args)
    try:
        result = pitcher_vulnerability(
            ctx, batter_id, date,
            pitchers_faced=pitchers_faced,
            pitcher_vuln=pitcher_vuln,
        )
        if result is not None:
            fired.append(result)
    except Exception as exc:
        logger.warning(
            "Signal pitcher_vulnerability failed for batter %d on %s: %s",
            batter_id, date, exc,
        )

    return fired


def load_batter_date_pitchers(conn: sqlite3.Connection) -> dict:
    """Load mapping of (batter_id, date) -> set of pitcher IDs faced.

    Used to supply the pitchers_faced argument to pitcher_vulnerability.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT game_date, batter, pitcher
        FROM statcast_staging
        WHERE events IS NOT NULL
    """)
    mapping = defaultdict(set)
    for r in cur.fetchall():
        mapping[(r["batter"], r["game_date"])].add(r["pitcher"])
    return dict(mapping)
