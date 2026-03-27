"""
mlb/signals/cold_streak_rebound.py — Detects batters likely to rebound from cold streaks.

Fires when a batter is in a cold streak but underlying production indicators
(hit tier, recent-game hit history) suggest the drought is unsustainable.
Uses the existing rebound_tier logic as a foundation and layers pitcher
vulnerability on top.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from mlb.signals.base import BaseSignal, ReasonLine, SignalResult


class ColdStreakRebound(BaseSignal):
    name = "cold_streak_rebound"
    description = "Batter in a cold streak with indicators suggesting imminent rebound"

    # Minimum cold streak length to even consider
    MIN_COLD_STREAK = -2

    def evaluate(
        self,
        batter: pd.Series,
        pitcher: Optional[pd.Series],
        matchup: Optional[pd.Series],
    ) -> SignalResult:
        streak = int(batter.get("current_streak", 0) or 0)
        if streak > self.MIN_COLD_STREAK:
            return self._not_fired()

        rebound_tier = batter.get("rebound_tier") or ""
        hit_tier = batter.get("hit_tier", "")
        season_hit_pct = float(batter.get("season_hit_pct", 0) or 0)

        # Must have a rebound tier or be Above Avg / Elite with a cold streak
        has_rebound = rebound_tier in ("Potential Rebound", "Likely Rebound", "Very Likely Rebound")
        is_good_hitter = hit_tier in ("Above Avg", "Elite")

        if not has_rebound and not (is_good_hitter and streak <= -3):
            return self._not_fired()

        # Build confidence from contributing factors
        reasons = []
        confidence_points = []

        # Factor 1: Streak severity (longer cold streak for a good hitter = higher rebound chance)
        streak_abs = abs(streak)
        reasons.append(ReasonLine(
            metric="Current streak",
            value=f"{streak} games without a hit",
            context=f"hit tier: {hit_tier}",
            direction="negative",
        ))
        if is_good_hitter:
            streak_score = min(streak_abs / 8.0, 1.0)  # caps at 8-game hitless
            confidence_points.append(("streak_severity", streak_score, 0.25))

        # Factor 2: Rebound tier
        tier_scores = {
            "Very Likely Rebound": 0.90,
            "Likely Rebound": 0.65,
            "Potential Rebound": 0.40,
        }
        tier_score = tier_scores.get(rebound_tier, 0.20)
        if rebound_tier:
            reasons.append(ReasonLine(
                metric="Rebound tier",
                value=rebound_tier,
                context=f"based on hit tier ({hit_tier}) and recent game pattern",
                direction="positive",
            ))
        confidence_points.append(("rebound_tier", tier_score, 0.35))

        # Factor 3: Season batting average (good hitters rebound more reliably)
        if season_hit_pct > 0:
            avg_score = min(season_hit_pct / 0.300, 1.0)
            reasons.append(ReasonLine(
                metric="Season batting avg",
                value=f".{int(season_hit_pct * 1000):03d}",
                context="higher avg = stronger rebound expectation",
                direction="positive" if season_hit_pct >= 0.250 else "neutral",
            ))
            confidence_points.append(("season_avg", avg_score, 0.15))

        # Factor 4: Pitcher vulnerability (if available)
        pitcher_boost = 0.0
        if pitcher is not None:
            p_avg_against = float(pitcher.get("avg_against", 0) or 0)
            if p_avg_against > 0.260:
                pitcher_boost = min((p_avg_against - 0.260) / 0.060, 1.0)
                reasons.append(ReasonLine(
                    metric="Opposing pitcher AVG against",
                    value=f".{int(p_avg_against * 1000):03d}",
                    context="above .260 = hittable",
                    direction="positive",
                ))
            confidence_points.append(("pitcher_vuln", pitcher_boost, 0.25))

        # Compute weighted confidence
        if not confidence_points:
            return self._not_fired()

        total_weight = sum(w for _, _, w in confidence_points)
        confidence = sum(score * weight for _, score, weight in confidence_points) / total_weight
        confidence = round(min(confidence, 0.95), 3)

        if confidence < 0.30:
            return self._not_fired()

        # Interpretation
        parts = [f"{batter.get('batter_name', 'Batter')} is {streak_abs} games into a hitless streak"]
        if is_good_hitter:
            parts.append(f"but is a {hit_tier} hitter (.{int(season_hit_pct * 1000):03d} season avg)")
        if rebound_tier:
            parts.append(f"classified as {rebound_tier}")
        if pitcher_boost > 0.3:
            parts.append("and faces a hittable pitcher today")
        interpretation = ", ".join(parts) + "."

        return SignalResult(
            fired=True,
            signal_type=self.name,
            confidence=confidence,
            headline=f"Cold streak rebound candidate ({rebound_tier or hit_tier})",
            reasons=reasons,
            interpretation=interpretation,
        )
