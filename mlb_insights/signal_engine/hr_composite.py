"""
mlb_insights/signal_engine/hr_composite.py -- HR composite scoring and ranking.

Combines calibrated HR probabilities with HR-specific signal boosts,
then z-score normalizes to a 0-100 scale.  Follows the same pattern
as composite.py but uses HR_SIGNAL_WEIGHTS and HR-specific features.
"""

import logging
from dataclasses import dataclass

import numpy as np

from mlb_insights.config import (
    HR_SIGNAL_WEIGHTS, ZSCORE_FLOOR, ZSCORE_CEIL, SCORE_MIN, SCORE_MAX,
    MIN_SIGNALS_FOR_BOOST, MIN_SINGLE_SIGNAL_CONFIDENCE,
)
from mlb_insights.signal_engine.signals import SignalResult

logger = logging.getLogger(__name__)


@dataclass
class HRScoredPlayer:
    """A single player's HR composite score for one date."""
    date: str
    player_id: int
    composite_raw: float
    hr_score: float          # normalized 0-100
    p_hr: float              # calibrated or raw p_hr_v2
    signals: list[SignalResult]
    hr_rank: int = 0
    # Feature values for display
    barrel_rate: float = 0.0
    fly_ball_rate: float = 0.0
    hard_hit_fly_rate: float = 0.0
    park_factor: float = 1.0
    # Actuals (filled by tracking)
    actual_hr: int | None = None

    @property
    def active_hr_signal_count(self) -> int:
        return len(self.signals)

    @property
    def top_hr_signal(self) -> str | None:
        if self.signals:
            return self.signals[0].signal_type
        return "base_hr_probability"

    @property
    def top_hr_reason(self) -> str | None:
        if self.signals:
            return self.signals[0].headline
        return f"P(HR) {self.p_hr*100:.1f}%"


def compute_hr_composite(
    p_hr: float,
    hr_signals: list[SignalResult],
) -> float:
    """Compute the raw HR composite score for a single player.

    composite = p_hr + sum(HR_SIGNAL_WEIGHTS[sig.type] * confidence)

    Args:
        p_hr: Calibrated or raw P(HR) from hr_features_v2.
        hr_signals: List of fired HR-specific signals.

    Returns:
        Raw HR composite score (not yet normalized).
    """
    base_score = p_hr

    signal_boost = 0.0
    for sig in hr_signals:
        weight = HR_SIGNAL_WEIGHTS.get(sig.signal_type, 0.05)
        signal_boost += weight * sig.confidence

    # Minimum signal gate: suppress noise from single weak signals
    n_signals = len(hr_signals)
    if n_signals < MIN_SIGNALS_FOR_BOOST:
        if n_signals == 0 or hr_signals[0].confidence < MIN_SINGLE_SIGNAL_CONFIDENCE:
            signal_boost = 0.0

    return base_score + signal_boost


def rank_hr_players(
    players: list[HRScoredPlayer],
    use_global_stats: bool = False,
    global_mean: float | None = None,
    global_std: float | None = None,
) -> list[HRScoredPlayer]:
    """Z-score normalize raw HR composite scores to 0-100 and assign ranks.

    When running a single date, normalization is within that date's cohort.
    When backfilling, pass use_global_stats=True with pre-computed mean/std
    for cross-date consistency.

    Args:
        players: List of HRScoredPlayer with composite_raw set.
        use_global_stats: If True, use provided global_mean/global_std.
        global_mean: Pre-computed mean of raw scores (for backfill).
        global_std: Pre-computed std of raw scores (for backfill).

    Returns:
        Same list, mutated with hr_score and hr_rank set.
    """
    if not players:
        return players

    raw_scores = np.array([p.composite_raw for p in players])

    if use_global_stats and global_mean is not None and global_std is not None:
        mean_score = global_mean
        std_score = global_std
    else:
        mean_score = float(np.mean(raw_scores))
        std_score = float(np.std(raw_scores))

    if std_score == 0:
        std_score = 1.0  # Avoid division by zero

    for p in players:
        z = (p.composite_raw - mean_score) / std_score
        # Scale z-score to 0-100: z of FLOOR -> 0, z of CEIL -> 100
        scaled = (z - ZSCORE_FLOOR) / (ZSCORE_CEIL - ZSCORE_FLOOR) * (SCORE_MAX - SCORE_MIN)
        p.hr_score = round(max(SCORE_MIN, min(SCORE_MAX, scaled)), 2)

    # Sort by score descending, assign ranks
    players.sort(key=lambda p: p.hr_score, reverse=True)
    for i, p in enumerate(players, 1):
        p.hr_rank = i

    if players:
        scores = [p.hr_score for p in players]
        logger.info(
            "Ranked %d HR players. Score range: [%.1f, %.1f], mean=%.1f",
            len(players), min(scores), max(scores), np.mean(scores),
        )

    return players


def rank_hr_players_by_date(
    all_players: list[HRScoredPlayer],
) -> list[HRScoredPlayer]:
    """Group HR players by date, normalize within each date, assign per-date ranks.

    Args:
        all_players: List of HRScoredPlayer across multiple dates.

    Returns:
        Same list, mutated with hr_score and hr_rank set per-date.
    """
    from collections import defaultdict

    by_date = defaultdict(list)
    for p in all_players:
        by_date[p.date].append(p)

    result = []
    for date in sorted(by_date.keys()):
        ranked = rank_hr_players(by_date[date])
        result.extend(ranked)

    return result


def rank_hr_players_global(
    all_players: list[HRScoredPlayer],
) -> list[HRScoredPlayer]:
    """Normalize using global stats across all dates, then rank per-date.

    Args:
        all_players: List of HRScoredPlayer across multiple dates.

    Returns:
        Same list, mutated with hr_score and hr_rank set.
    """
    from collections import defaultdict

    if not all_players:
        return all_players

    raw_scores = np.array([p.composite_raw for p in all_players])
    global_mean = float(np.mean(raw_scores))
    global_std = float(np.std(raw_scores))

    logger.info(
        "HR global normalization: mean=%.4f, std=%.4f over %d entries.",
        global_mean, global_std, len(all_players),
    )

    by_date = defaultdict(list)
    for p in all_players:
        by_date[p.date].append(p)

    result = []
    for date in sorted(by_date.keys()):
        ranked = rank_hr_players(
            by_date[date],
            use_global_stats=True,
            global_mean=global_mean,
            global_std=global_std,
        )
        result.extend(ranked)

    return result
