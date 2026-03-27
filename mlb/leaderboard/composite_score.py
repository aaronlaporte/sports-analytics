"""
mlb/leaderboard/composite_score.py — Compute daily composite scores for each batter.

Combines P(1+ hit), P(2+ hits), P(HR), and active signal boosts into
a single daily_score used for leaderboard ranking.
"""

from __future__ import annotations

import pandas as pd

from mlb.signals.base import SignalResult

# Default weights for composite score components
W_1HIT = 0.40
W_2HIT = 0.30
W_HR = 0.15
SIGNAL_BOOST_CAP = 0.15


def signal_boost(signals: list[SignalResult]) -> float:
    """Compute a signal-based boost from 0.0 to SIGNAL_BOOST_CAP.

    - 1 active signal: +0.03
    - 2 active signals: +0.06
    - 3+ active signals: +0.10
    - Any signal with confidence > 0.80: additional +0.05
    """
    if not signals:
        return 0.0

    count = len(signals)
    if count >= 3:
        boost = 0.10
    elif count == 2:
        boost = 0.06
    else:
        boost = 0.03

    # High-confidence bonus
    if any(s.confidence > 0.80 for s in signals):
        boost += 0.05

    return min(boost, SIGNAL_BOOST_CAP)


def compute_daily_score(
    p_1hit: float,
    p_2hit: float,
    p_hr: float,
    signals: list[SignalResult] | None = None,
) -> float:
    """Compute the composite daily score for one batter.

    Returns a float typically in the 0.3 - 0.95 range.
    """
    base = (W_1HIT * p_1hit) + (W_2HIT * p_2hit) + (W_HR * p_hr)
    boost = signal_boost(signals or [])
    return round(min(base + boost, 1.0), 4)


def score_all_batters(
    batters_df: pd.DataFrame,
    all_signals: dict[int, list[SignalResult]],
) -> pd.DataFrame:
    """Add daily_score, active_signal_count, top_signal, top_reason to batters_df.

    Expects batters_df to have: batter_id, p_1hit (or model_prob), p_2hit, p_hr.
    """
    scores = []

    for _, row in batters_df.iterrows():
        batter_id = int(row.get("batter_id", row.get("batter", 0)))
        p1 = float(row.get("p_1hit", row.get("model_prob", 0)) or 0)
        p2 = float(row.get("p_2hit", 0) or 0)
        p_hr = float(row.get("p_hr", 0) or 0)

        signals = all_signals.get(batter_id, [])
        daily = compute_daily_score(p1, p2, p_hr, signals)

        # Pick the top signal by confidence
        top_signal = ""
        top_reason = ""
        if signals:
            best = max(signals, key=lambda s: s.confidence)
            top_signal = best.signal_type
            top_reason = best.headline

        scores.append({
            "batter_id": batter_id,
            "daily_score": daily,
            "active_signal_count": len(signals),
            "top_signal": top_signal,
            "top_reason": top_reason,
        })

    scores_df = pd.DataFrame(scores)
    return batters_df.merge(scores_df, on="batter_id", how="left")
