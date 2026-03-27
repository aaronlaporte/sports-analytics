"""
mlb/signals/hot_streak_acceleration.py — Detects batters on hot streaks facing vulnerable pitchers.

Fires when a batter is on a hot streak AND today's matchup conditions suggest
continued production (weak pitcher, favorable matchup history).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from mlb.signals.base import BaseSignal, ReasonLine, SignalResult


class HotStreakAcceleration(BaseSignal):
    name = "hot_streak_acceleration"
    description = "Batter on a hot streak facing conditions that favor continued production"

    MIN_HOT_STREAK = 4

    def evaluate(
        self,
        batter: pd.Series,
        pitcher: Optional[pd.Series],
        matchup: Optional[pd.Series],
    ) -> SignalResult:
        streak = int(batter.get("current_streak", 0) or 0)
        if streak < self.MIN_HOT_STREAK:
            return self._not_fired()

        reasons = []
        confidence_points = []

        # Factor 1: Streak length
        streak_score = min(streak / 10.0, 1.0)
        reasons.append(ReasonLine(
            metric="Current hot streak",
            value=f"{streak} consecutive games with a hit",
            context="longer streaks indicate sustained form",
            direction="positive",
        ))
        confidence_points.append(("streak_length", streak_score, 0.30))

        # Factor 2: Recent production volume
        hits_pg_5 = float(batter.get("hits_pg_5", 0) or 0)
        if hits_pg_5 > 0:
            recent_score = min(hits_pg_5 / 10.0, 1.0)  # 10 hits in 5 games = max
            reasons.append(ReasonLine(
                metric="Hits in last 5 games",
                value=str(int(hits_pg_5)),
                context="volume of recent production",
                direction="positive" if hits_pg_5 >= 6 else "neutral",
            ))
            confidence_points.append(("recent_volume", recent_score, 0.20))

        # Factor 3: Pitcher vulnerability
        if pitcher is not None:
            p_avg_against = float(pitcher.get("avg_against", 0) or 0)
            p_hits_allowed = int(pitcher.get("hits_allowed", 0) or 0)
            p_bf = int(pitcher.get("batters_faced", 0) or 0)

            if p_avg_against > 0.250:
                vuln_score = min((p_avg_against - 0.250) / 0.070, 1.0)
                reasons.append(ReasonLine(
                    metric="Opposing pitcher AVG against",
                    value=f".{int(p_avg_against * 1000):03d}",
                    context="above .250 = favorable",
                    direction="positive",
                ))
                confidence_points.append(("pitcher_vuln", vuln_score, 0.25))
            else:
                # Tough pitcher dampens the signal
                reasons.append(ReasonLine(
                    metric="Opposing pitcher AVG against",
                    value=f".{int(p_avg_against * 1000):03d}",
                    context="below .250 = tough matchup",
                    direction="negative",
                ))
                confidence_points.append(("pitcher_vuln", 0.15, 0.25))

        # Factor 4: Matchup history (if available)
        if matchup is not None:
            m_pa = int(matchup.get("pa", 0) or 0)
            m_avg = float(matchup.get("avg", 0) or 0)
            if m_pa >= 5:
                matchup_score = min(m_avg / 0.350, 1.0)
                reasons.append(ReasonLine(
                    metric="Career AVG vs this pitcher",
                    value=f".{int(m_avg * 1000):03d}",
                    context=f"in {m_pa} PA",
                    direction="positive" if m_avg >= 0.300 else "neutral",
                ))
                confidence_points.append(("matchup_history", matchup_score, 0.25))

        if not confidence_points:
            return self._not_fired()

        total_weight = sum(w for _, _, w in confidence_points)
        confidence = sum(score * weight for _, score, weight in confidence_points) / total_weight
        confidence = round(min(confidence, 0.95), 3)

        if confidence < 0.35:
            return self._not_fired()

        batter_name = batter.get("batter_name", "Batter")
        interpretation = (
            f"{batter_name} has hit safely in {streak} straight games"
            f" and conditions today favor continued production."
        )

        return SignalResult(
            fired=True,
            signal_type=self.name,
            confidence=confidence,
            headline=f"Hot streak ({streak} games) with favorable matchup",
            reasons=reasons,
            interpretation=interpretation,
        )
