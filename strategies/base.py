"""Base class for all strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from core.models import PriceSeries, Regime, Signal


class Strategy(ABC):
    """
    All strategies must:
      1. Accept a PriceSeries with indicators already computed
      2. Return a Signal (with direction, strength, SL/TP) or None
      3. Be stateless — no internal memory between calls
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate(
        self,
        series: PriceSeries,
        features: dict[str, float],
        regime: Regime,
    ) -> Optional[Signal]:
        """
        Evaluate the strategy on current data.
        Returns a Signal if conditions are met, None otherwise.
        """
        ...
