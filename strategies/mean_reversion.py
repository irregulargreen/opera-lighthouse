"""
Mean reversion strategy — profits from overextended moves snapping back.

Exploits: short-term overreaction, bid-ask bounce, liquidity provision dynamics.

Entry: price at Bollinger Band extremes + RSI divergence + ranging regime
Exit: reversion to mean (BB middle / SMA20) or tight stop beyond the band

Typical stats: ~58% win rate, 0.8:1 reward:risk ratio.
Profitable because of high hit rate on small moves.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from core.instrument_params import get_params
from core.models import Direction, PriceSeries, Regime, Signal, SignalSource
from strategies.base import Strategy


class MeanReversion(Strategy):

    @property
    def name(self) -> str:
        return "mean_reversion"

    def evaluate(
        self,
        series: PriceSeries,
        features: dict[str, float],
        regime: Regime,
    ) -> Optional[Signal]:
        if not series.candles or len(series.candles) < 50:
            return None

        params = get_params(series.instrument)

        if not params.enable_reversion:
            return None

        if regime != Regime.RANGING:
            return None

        price = series.closes[-1]
        rsi = features.get("rsi", 50.0)
        bb_pct = features.get("bb_pct", 0.5)
        adx = features.get("adx", 25.0)
        atr_ratio = features.get("atr_ratio", 1.0)
        dist_sma20 = features.get("dist_sma20", 0.0)
        htf_bias = features.get("htf_bias", 0.0)

        if adx > params.adx_reversion_max:
            return None

        if atr_ratio > 1.3:
            return None

        # Don't fade a strong higher-timeframe trend
        if abs(htf_bias) > 0.5:
            return None

        raw_atr = self._get_raw_atr(series)
        if raw_atr <= 0:
            return None

        is_long = bb_pct < 0.05 and rsi < 30 and dist_sma20 < -1.5
        is_short = bb_pct > 0.95 and rsi > 70 and dist_sma20 > 1.5

        if not (is_long or is_short):
            return None

        # ── Strength calculation ────────────────────────────────────────────

        strength = 0.0

        if is_long:
            bb_score = min(0.4, (0.05 - bb_pct) * 8.0)
        else:
            bb_score = min(0.4, (bb_pct - 0.95) * 8.0)
        strength += max(0, bb_score)

        if is_long:
            rsi_score = min(0.3, (30 - rsi) / 80)
        else:
            rsi_score = min(0.3, (rsi - 70) / 80)
        strength += max(0, rsi_score)

        dist_score = min(0.3, abs(dist_sma20) * 0.1)
        strength += dist_score

        strength = round(min(1.0, strength), 4)

        # ── Stop and target ─────────────────────────────────────────────────

        bb_mid = self._get_bb_mid(series)

        if is_long:
            direction = Direction.LONG
            stop_loss = price - params.atr_stop_reversion * raw_atr
            take_profit = bb_mid if bb_mid > price else price + params.atr_target_reversion * raw_atr
        else:
            direction = Direction.SHORT
            stop_loss = price + params.atr_stop_reversion * raw_atr
            take_profit = bb_mid if bb_mid < price else price - params.atr_target_reversion * raw_atr

        confidence = min(1.0, strength * (1.0 - adx / 50))

        return Signal(
            instrument=series.instrument,
            source=SignalSource.REVERSION,
            direction=direction,
            strength=strength,
            confidence=round(confidence, 4),
            entry_price=price,
            stop_loss=round(stop_loss, 6),
            take_profit=round(take_profit, 6),
            atr=raw_atr,
            regime=regime,
            features=features,
            reasoning=(
                f"Reversion {'long' if is_long else 'short'}: "
                f"BB%={bb_pct:.2f} RSI={rsi:.0f} ADX={adx:.0f} "
                f"dist_SMA20={dist_sma20:+.1f}ATR HTF={features.get('htf_bias', 0):+.2f}"
            ),
        )

    @staticmethod
    def _get_raw_atr(series: PriceSeries) -> float:
        if series.atr_14:
            valid = [v for v in series.atr_14[-5:] if v is not None and not np.isnan(v)]
            if valid:
                return float(valid[-1])
        return 0.0

    @staticmethod
    def _get_bb_mid(series: PriceSeries) -> float:
        if series.bb_middle:
            valid = [v for v in series.bb_middle[-5:] if v is not None and not np.isnan(v)]
            if valid:
                return float(valid[-1])
        return series.closes[-1]
