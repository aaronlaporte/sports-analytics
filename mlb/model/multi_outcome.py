"""
mlb/model/multi_outcome.py — Extend the beta-binomial hit model to P(2+ hits) and P(HR).

Uses the same per-PA hit rate from the existing model but applies binomial
expansion to compute multi-hit and home run probabilities.
"""

from __future__ import annotations

from math import comb as binomial_coeff
from typing import Optional

import pandas as pd


def p_exactly_k_hits(p_hit_per_pa: float, expected_pa: float, k: int) -> float:
    """Binomial probability of exactly k hits in n plate appearances.

    P(X = k) = C(n, k) * p^k * (1-p)^(n-k)

    Uses floor of expected_pa as n (can't have fractional PA).
    """
    n = max(int(expected_pa), 1)
    if k > n or k < 0:
        return 0.0
    return binomial_coeff(n, k) * (p_hit_per_pa ** k) * ((1 - p_hit_per_pa) ** (n - k))


def p_2plus_hits(p_hit_per_pa: float, expected_pa: float) -> float:
    """P(>=2 hits) = 1 - P(0 hits) - P(1 hit)."""
    p0 = p_exactly_k_hits(p_hit_per_pa, expected_pa, 0)
    p1 = p_exactly_k_hits(p_hit_per_pa, expected_pa, 1)
    return round(max(1.0 - p0 - p1, 0.0), 4)


def p_home_run(batter: pd.Series, pitcher: Optional[pd.Series] = None) -> float:
    """Estimate P(>=1 HR) for a batter in today's game.

    Simple shrinkage model:
        hr_rate = (season_hr + league_hr_rate * prior) / (season_ab + prior)
        P(HR) = 1 - (1 - hr_rate)^expected_pa

    If pitcher data is available, blend with pitcher HR-allowed rate.
    """
    season_hr = int(batter.get("hr", 0) or 0)
    season_ab = int(batter.get("ab", 0) or 0)
    expected_pa = float(batter.get("pa", 3.5) or 3.5)
    expected_pa = min(max(expected_pa, 3.0), 5.0)

    # League average HR rate per AB (~3.2% in recent seasons)
    league_hr_rate = 0.032
    prior_ab = 200

    if season_ab > 0:
        batter_hr_rate = (season_hr + league_hr_rate * prior_ab) / (season_ab + prior_ab)
    else:
        batter_hr_rate = league_hr_rate

    # Blend with pitcher HR-allowed rate if available
    if pitcher is not None:
        p_hr_allowed = int(pitcher.get("hr_allowed", 0) or 0)
        p_bf = int(pitcher.get("batters_faced", 0) or 0)
        if p_bf > 30:
            pitcher_hr_rate = (p_hr_allowed + league_hr_rate * 100) / (p_bf + 100)
            # 60/40 blend: batter talent weighted more than pitcher
            blended_rate = 0.60 * batter_hr_rate + 0.40 * pitcher_hr_rate
        else:
            blended_rate = batter_hr_rate
    else:
        blended_rate = batter_hr_rate

    p_no_hr = (1 - blended_rate) ** expected_pa
    return round(max(1.0 - p_no_hr, 0.0), 4)


def compute_multi_outcomes(
    batters_df: pd.DataFrame,
    pitcher_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add p_2hit and p_hr columns to a batters DataFrame.

    Expects batters_df to have: model_prob (P(1+ hit) per PA rate is derived from it),
    pa (expected PA), and optionally opp_pitcher_id for pitcher lookup.
    """
    results = []

    for _, batter in batters_df.iterrows():
        model_prob = float(batter.get("model_prob", batter.get("p_hit", 0)) or 0)
        expected_pa = float(batter.get("expected_pa", batter.get("pa", 3.5)) or 3.5)
        expected_pa = min(max(expected_pa, 3.0), 5.0)

        # Reverse-engineer per-PA hit rate from P(1+ hit)
        # P(1+ hit) = 1 - (1-p)^n  =>  p = 1 - (1 - P(1+hit))^(1/n)
        if model_prob > 0 and model_prob < 1:
            p_per_pa = 1 - (1 - model_prob) ** (1 / expected_pa)
        else:
            p_per_pa = 0.0

        p2 = p_2plus_hits(p_per_pa, expected_pa)

        # HR probability
        opp_pitcher_id = batter.get("opp_pitcher_id")
        pitcher_row = None
        if pitcher_df is not None and opp_pitcher_id is not None:
            p_match = pitcher_df[pitcher_df["pitcher_id"] == int(opp_pitcher_id)]
            if not p_match.empty:
                pitcher_row = p_match.iloc[-1]

        p_hr = p_home_run(batter, pitcher_row)

        results.append({
            "batter_id": int(batter.get("batter_id", batter.get("batter", 0))),
            "p_2hit": p2,
            "p_hr": p_hr,
        })

    return pd.DataFrame(results)
