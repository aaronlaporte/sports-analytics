"""
mlb_insights/signal_engine/composite.py -- Composite scoring and player ranking.

Extracted from phase2b_rebuild.py Step 7.  Combines calibrated probabilities
with signal-based boosts, then z-score normalizes to a 0-100 scale.
"""

import logging
from dataclasses import dataclass

import numpy as np

from mlb_insights.config import (
    SIGNAL_WEIGHTS, ZSCORE_FLOOR, ZSCORE_CEIL, SCORE_MIN, SCORE_MAX,
    MIN_SIGNALS_FOR_BOOST, MIN_SINGLE_SIGNAL_CONFIDENCE,
)
from mlb_insights.signal_engine.signals import SignalResult

logger = logging.getLogger(__name__)


@dataclass
class ScoredPlayer:
    """A single player's composite score for one date."""
    date: str
    player_id: int
    composite_raw: float
    daily_score: float
    cal_p1hit: float
    cal_p2hit: float
    cal_phr: float
    signals: list[SignalResult]
    daily_rank: int = 0

    # Optional actuals (filled by tracking)
    actual_hits: int | None = None
    actual_hr: int | None = None
    actual_pa: int | None = None

    @property
    def active_signal_count(self) -> int:
        return len(self.signals)

    @property
    def top_signal(self) -> str | None:
        if self.signals:
            return self.signals[0].signal_type
        return "base_probability"

    @property
    def top_reason(self) -> str | None:
        if self.signals:
            return self.signals[0].headline
        # Explain why this player ranks high despite no signals
        return (
            f"High base probability — P(1H) {self.cal_p1hit*100:.0f}%, "
            f"P(2H) {self.cal_p2hit*100:.0f}%, "
            f"P(HR) {self.cal_phr*100:.0f}%"
        )


def compute_composite_score(
    cal_p1hit: float,
    cal_p2hit: float,
    cal_phr: float,
    active_signals: list[SignalResult],
) -> float:
    """Compute the raw composite score for a single player.

    composite = calibrated_p_1hit + sum(signal_weight * confidence)

    Args:
        cal_p1hit: Calibrated P(1+ hit).
        cal_p2hit: Calibrated P(2+ hit).  (reserved for future weighting)
        cal_phr: Calibrated P(HR).  (reserved for future weighting)
        active_signals: List of fired signals.

    Returns:
        Raw composite score (not yet normalized).
    """
    base_score = cal_p1hit

    signal_boost = 0.0
    for sig in active_signals:
        weight = SIGNAL_WEIGHTS.get(sig.signal_type, 0.05)
        signal_boost += weight * sig.confidence

    # Minimum signal gate: suppress noise from single weak signals
    n_signals = len(active_signals)
    if n_signals < MIN_SIGNALS_FOR_BOOST:
        if n_signals == 0 or active_signals[0].confidence < MIN_SINGLE_SIGNAL_CONFIDENCE:
            signal_boost = 0.0

    return base_score + signal_boost


def rank_players(
    players: list[ScoredPlayer],
    use_global_stats: bool = False,
    global_mean: float | None = None,
    global_std: float | None = None,
) -> list[ScoredPlayer]:
    """Z-score normalize raw composite scores to 0-100 and assign ranks.

    When running a single date, normalization is within that date's cohort.
    When backfilling, pass use_global_stats=True with pre-computed mean/std
    for cross-date consistency.

    Args:
        players: List of ScoredPlayer with composite_raw set.
        use_global_stats: If True, use provided global_mean/global_std.
        global_mean: Pre-computed mean of raw scores (for backfill).
        global_std: Pre-computed std of raw scores (for backfill).

    Returns:
        Same list, mutated with daily_score and daily_rank set.
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
        p.daily_score = round(max(SCORE_MIN, min(SCORE_MAX, scaled)), 2)

    # Sort by score descending, assign ranks
    players.sort(key=lambda p: p.daily_score, reverse=True)
    for i, p in enumerate(players, 1):
        p.daily_rank = i

    if players:
        scores = [p.daily_score for p in players]
        logger.info(
            "Ranked %d players. Score range: [%.1f, %.1f], mean=%.1f",
            len(players), min(scores), max(scores), np.mean(scores),
        )

    return players


def rank_players_by_date(
    all_players: list[ScoredPlayer],
) -> list[ScoredPlayer]:
    """Group players by date, normalize within each date, assign per-date ranks.

    This is the standard mode for backfill: each date gets its own normalization.

    Args:
        all_players: List of ScoredPlayer across multiple dates.

    Returns:
        Same list, mutated with daily_score and daily_rank set per-date.
    """
    from collections import defaultdict

    by_date = defaultdict(list)
    for p in all_players:
        by_date[p.date].append(p)

    result = []
    for date in sorted(by_date.keys()):
        ranked = rank_players(by_date[date])
        result.extend(ranked)

    return result


def rank_players_global(
    all_players: list[ScoredPlayer],
) -> list[ScoredPlayer]:
    """Normalize using global stats across all dates, then rank per-date.

    This produces more comparable scores across dates (as phase2b did).

    Args:
        all_players: List of ScoredPlayer across multiple dates.

    Returns:
        Same list, mutated with daily_score and daily_rank set.
    """
    from collections import defaultdict

    if not all_players:
        return all_players

    raw_scores = np.array([p.composite_raw for p in all_players])
    global_mean = float(np.mean(raw_scores))
    global_std = float(np.std(raw_scores))

    logger.info(
        "Global normalization: mean=%.4f, std=%.4f over %d entries.",
        global_mean, global_std, len(all_players),
    )

    by_date = defaultdict(list)
    for p in all_players:
        by_date[p.date].append(p)

    result = []
    for date in sorted(by_date.keys()):
        ranked = rank_players(
            by_date[date],
            use_global_stats=True,
            global_mean=global_mean,
            global_std=global_std,
        )
        result.extend(ranked)

    return result
