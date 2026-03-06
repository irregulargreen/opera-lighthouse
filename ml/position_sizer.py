"""
Adaptive position sizing — how much to risk on each trade.

Fixed sizing (risk X% per trade) is the default.
As data accumulates, the sizer learns to:
  - Size up when conditions match historically profitable setups
  - Size down when conditions match historically unprofitable ones
  - Respect hard risk limits regardless of model confidence

Core formula: risk = base_risk × kelly_fraction × confidence_multiplier

Where confidence_multiplier comes from:
  - Cold start: 1.0 (use base risk)
  - After ML: scaled by P(win) from signal combiner and historical payoff ratio
"""
from __future__ import annotations

from loguru import logger

from core.config import config
from core.models import Direction, Signal


class PositionSizer:
    """
    Determines position size in units, given a signal and account state.

    Uses ATR-based risk calculation:
      risk_dollars = capital × max_risk_per_trade
      risk_in_price = |entry - stop_loss|
      units = risk_dollars / risk_in_price
    """

    def calculate(
        self,
        signal: Signal,
        capital: float,
        win_prob: float = 0.5,
        avg_win_loss_ratio: float = 1.5,
        pip_value: float = 0.0001,
    ) -> dict:
        """
        Calculate position size.

        Returns dict with:
          units: number of currency units to trade
          risk_usdc: dollar amount at risk
          kelly_fraction: effective Kelly fraction used
        """
        base_risk = config.trading.max_risk_per_trade

        # ── Kelly-based adjustment ──────────────────────────────────────────
        # Full Kelly: f* = (p × b - q) / b
        # where p = win_prob, q = 1-p, b = avg_win/avg_loss ratio
        b = max(0.5, avg_win_loss_ratio)
        p = win_prob
        q = 1.0 - p

        kelly = (p * b - q) / b if b > 0 else 0.0
        kelly = max(0.0, kelly)

        # Apply fractional Kelly
        effective_kelly = kelly * config.trading.kelly_fraction

        # Scale risk by Kelly (but never above base_risk × 1.5 or below × 0.3)
        if effective_kelly > 0:
            risk_multiplier = min(1.5, max(0.3, effective_kelly / base_risk))
        else:
            risk_multiplier = 0.3  # Minimum sizing

        adjusted_risk = base_risk * risk_multiplier
        risk_dollars = capital * adjusted_risk

        # ── Convert to units ────────────────────────────────────────────────
        risk_in_price = abs(signal.entry_price - signal.stop_loss)
        if risk_in_price <= 0:
            logger.warning("[Sizer] Zero risk distance — cannot size position")
            return {"units": 0, "risk_usdc": 0, "kelly_fraction": 0}

        units = risk_dollars / risk_in_price

        # Round to reasonable lot sizes
        if units > 100:
            units = round(units / 100) * 100
        units = max(1, int(units))

        # Negative units for short positions
        if signal.direction == Direction.SHORT:
            units = -units

        risk_pips = risk_in_price / pip_value if pip_value > 0 else 0

        return {
            "units": units,
            "risk_usdc": round(risk_dollars, 2),
            "kelly_fraction": round(effective_kelly, 4),
            "risk_multiplier": round(risk_multiplier, 3),
            "risk_pips": round(risk_pips, 1),
            "win_prob_used": round(win_prob, 4),
        }
