"""
mlb/signals/engine.py — Evaluates all registered signals for each batter.

The SignalEngine loads all signal implementations, runs them against
batter/pitcher/matchup data, and returns collected results.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

import pandas as pd

from mlb.signals.base import BaseSignal, SignalResult
from mlb.signals.cold_streak_rebound import ColdStreakRebound
from mlb.signals.hot_streak_acceleration import HotStreakAcceleration
from mlb.signals.pitcher_vulnerability import PitcherVulnerability


# All registered signals — add new ones here as they're built
SIGNALS: list[BaseSignal] = [
    ColdStreakRebound(),
    HotStreakAcceleration(),
    PitcherVulnerability(),
]


class SignalEngine:
    """Evaluates all signals for a single batter against today's matchup."""

    def __init__(self, signals: list[BaseSignal] | None = None):
        self.signals = signals or SIGNALS

    def evaluate_batter(
        self,
        batter: pd.Series,
        pitcher: Optional[pd.Series] = None,
        matchup: Optional[pd.Series] = None,
    ) -> list[SignalResult]:
        """Run all signals for one batter. Returns only fired signals."""
        fired = []
        for signal in self.signals:
            result = signal.evaluate(batter, pitcher, matchup)
            if result.fired:
                fired.append(result)
        return fired

    def evaluate_all_batters(
        self,
        batters_df: pd.DataFrame,
        pitcher_df: Optional[pd.DataFrame] = None,
        matchup_df: Optional[pd.DataFrame] = None,
    ) -> dict[int, list[SignalResult]]:
        """Run all signals for every batter in the DataFrame.

        Args:
            batters_df:  DataFrame with one row per batter (most recent stats).
                         Must have 'batter_id' column.
            pitcher_df:  DataFrame with pitcher stats. Looked up by pitcher_id
                         if batter row contains 'opp_pitcher_id'.
            matchup_df:  DataFrame with matchup stats. Looked up by
                         (batter_id, pitcher_id) pair.

        Returns:
            Dict mapping batter_id -> list of fired SignalResults.
        """
        results: dict[int, list[SignalResult]] = {}

        for _, batter in batters_df.iterrows():
            batter_id = int(batter.get("batter_id", batter.get("batter", 0)))
            opp_pitcher_id = batter.get("opp_pitcher_id")

            # Look up pitcher row
            pitcher_row = None
            if pitcher_df is not None and opp_pitcher_id is not None:
                p_match = pitcher_df[pitcher_df["pitcher_id"] == int(opp_pitcher_id)]
                if not p_match.empty:
                    pitcher_row = p_match.iloc[-1]  # most recent

            # Look up matchup row
            matchup_row = None
            if matchup_df is not None and opp_pitcher_id is not None:
                m_match = matchup_df[
                    (matchup_df["batter_id"] == batter_id)
                    & (matchup_df["pitcher_id"] == int(opp_pitcher_id))
                ]
                if not m_match.empty:
                    matchup_row = m_match.iloc[-1]

            fired = self.evaluate_batter(batter, pitcher_row, matchup_row)
            if fired:
                results[batter_id] = fired

        return results


def signals_to_db_rows(
    signal_date: str,
    all_signals: dict[int, list[SignalResult]],
) -> list[tuple]:
    """Convert signal results to rows for the daily_signals table."""
    rows = []
    for player_id, signals in all_signals.items():
        for sig in signals:
            reasons_json = json.dumps(
                [asdict(r) for r in sig.reasons],
                ensure_ascii=False,
            )
            rows.append((
                signal_date,
                player_id,
                sig.signal_type,
                sig.confidence,
                sig.headline,
                reasons_json,
                sig.interpretation,
            ))
    return rows
