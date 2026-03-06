"""
Risk management — the layer that keeps you in the game.

Rules:
  1. Max positions: hard limit on concurrent open trades
  2. Correlation guard: no more than N positions in correlated pairs
  3. Drawdown circuit breaker: pause trading if equity drops >X%
  4. Daily loss limit: stop trading for the day if losses exceed threshold
  5. Per-instrument limit: max 1 open position per instrument
"""
from __future__ import annotations

from loguru import logger

from core.config import config
from core.models import Position, Signal


# Correlated pair groups — positions in the same group are correlated
CORRELATION_GROUPS = {
    "usd_long": ["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "XAU_USD"],
    "usd_short": ["USD_JPY", "USD_CHF", "USD_CAD"],
    "eur": ["EUR_USD", "EUR_GBP", "EUR_JPY"],
    "gbp": ["GBP_USD", "EUR_GBP", "GBP_JPY"],
    "jpy": ["USD_JPY", "EUR_JPY", "GBP_JPY"],
    "chf": ["USD_CHF", "EUR_CHF", "GBP_CHF"],
    "commodity": ["AUD_USD", "NZD_USD", "USD_CAD", "XAU_USD"],
}


class RiskManager:
    """Pre-trade risk checks and portfolio-level controls."""

    def __init__(self):
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._circuit_breaker_active: bool = False
        self._peak_equity: float = config.trading.total_capital

    def check(
        self,
        signal: Signal,
        open_positions: dict[str, Position],
        current_equity: float,
    ) -> tuple[bool, str]:
        """
        Run all risk checks. Returns (allowed, reason).
        """
        # Circuit breaker
        if self._circuit_breaker_active:
            return False, "circuit breaker active — trading paused"

        # Drawdown check
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        drawdown = (self._peak_equity - current_equity) / self._peak_equity
        if drawdown > 0.10:
            self._circuit_breaker_active = True
            logger.warning(
                f"[Risk] CIRCUIT BREAKER: drawdown {drawdown:.1%} exceeds 10%. "
                f"Peak: ${self._peak_equity:.2f} Current: ${current_equity:.2f}"
            )
            return False, f"drawdown {drawdown:.1%} > 10%"

        # Daily loss limit (3% of capital)
        daily_limit = config.trading.total_capital * 0.03
        if self._daily_pnl < -daily_limit:
            return False, f"daily loss ${self._daily_pnl:.2f} exceeds limit ${-daily_limit:.2f}"

        # Max positions
        n_open = len(open_positions)
        if n_open >= config.trading.max_open_positions:
            return False, f"max positions reached ({n_open})"

        # Per-instrument limit
        instrument_count = sum(
            1 for p in open_positions.values()
            if p.instrument == signal.instrument
        )
        if instrument_count >= 1:
            return False, f"already have position in {signal.instrument}"

        # Correlation guard
        correlated_count = self._count_correlated(signal.instrument, open_positions)
        if correlated_count >= config.trading.max_correlation_exposure:
            return False, (
                f"correlation limit: {correlated_count} correlated positions "
                f"for {signal.instrument}"
            )

        # Direction conflict: don't go long and short on same pair simultaneously
        for pos in open_positions.values():
            if pos.instrument == signal.instrument and pos.direction != signal.direction:
                return False, f"conflicting direction on {signal.instrument}"

        return True, "all checks passed"

    def record_daily_pnl(self, pnl: float):
        self._daily_pnl += pnl
        self._daily_trades += 1

    def reset_daily(self):
        self._daily_pnl = 0.0
        self._daily_trades = 0
        if self._circuit_breaker_active:
            logger.info("[Risk] Circuit breaker reset for new day")
            self._circuit_breaker_active = False

    def reset_circuit_breaker(self):
        self._circuit_breaker_active = False
        logger.info("[Risk] Circuit breaker manually reset")

    @staticmethod
    def _count_correlated(instrument: str, positions: dict[str, Position]) -> int:
        """Count how many open positions are correlated with this instrument."""
        groups = []
        for group_name, members in CORRELATION_GROUPS.items():
            if instrument in members:
                groups.append(group_name)

        if not groups:
            return 0

        correlated = set()
        for pos in positions.values():
            for group_name in groups:
                if pos.instrument in CORRELATION_GROUPS.get(group_name, []):
                    correlated.add(pos.trade_id)

        return len(correlated)
