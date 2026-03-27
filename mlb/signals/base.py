"""
mlb/signals/base.py — Base classes for the signal detection system.

Every signal inherits from BaseSignal and implements evaluate().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class ReasonLine:
    """One supporting data point in a signal explanation."""
    metric: str         # e.g., "Hard hit rate (last 5 games)"
    value: str          # e.g., "52%"
    context: str        # e.g., "vs season avg 41%"
    direction: str      # "positive" | "negative" | "neutral"


@dataclass
class SignalResult:
    """Output of a single signal evaluation for one batter."""
    fired: bool
    signal_type: str
    confidence: float               # 0.0 - 1.0
    headline: str                   # One-line summary
    reasons: list[ReasonLine] = field(default_factory=list)
    interpretation: str = ""        # Plain-english conclusion


class BaseSignal:
    """Abstract base for all signal implementations."""

    name: str = ""
    description: str = ""

    def evaluate(
        self,
        batter: pd.Series,
        pitcher: Optional[pd.Series],
        matchup: Optional[pd.Series],
    ) -> SignalResult:
        """Evaluate whether this signal fires for the given batter/pitcher/matchup.

        Args:
            batter:  Row from batter_stats (most recent game date for this batter).
            pitcher: Row from pitcher_stats for today's opposing pitcher (may be None).
            matchup: Row from matchup_stats for this batter-pitcher pair (may be None).

        Returns:
            SignalResult with fired=True/False, confidence, reasons, and interpretation.
        """
        raise NotImplementedError

    def _not_fired(self) -> SignalResult:
        """Convenience: return a non-fired result."""
        return SignalResult(
            fired=False,
            signal_type=self.name,
            confidence=0.0,
            headline="",
            reasons=[],
            interpretation="",
        )
