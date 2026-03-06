"""
Breakout strategy — catches the start of new trends from consolidation.

Exploits: volatility compression followed by expansion (squeeze → breakout),
          herd behavior when key levels break, stop-hunting dynamics.

Entry: Donchian channel breakout + Bollinger squeeze + volume confirmation
Exit: trailing stop at 2× ATR (captures trend continuation)

Typical stats: ~35% win rate, 3:1+ reward:risk ratio.
Low win rate but large winners compensate.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from core.instrument_params import get_params
from core.models import Direction, PriceSeries, Regime, Signal, SignalSource
from strategies.base import Strategy


class Breakout(Strategy):

    @property
    def name(self) -> str:
        return "breakout"

    def evaluate(
        self,
        series: PriceSeries,
        features: dict[str, float],
        regime: Regime,
    ) -> Optional[Signal]:
        if not series.candles or len(series.candles) < 50:
            return None

        params = get_params(series.instrument)
        price = series.closes[-1]
        prev_price = series.closes[-2]
        bb_width = features.get("bb_width", 0.02)
        adx = features.get("adx", 20.0)
        vol_ratio = features.get("volume_ratio", 1.0)
        rsi = features.get("rsi", 50.0)
        dc_pct = features.get("dc_pct", 0.5)
        htf_bias = features.get("htf_bias", 0.0)

        raw_atr = self._get_raw_atr(series)
        if raw_atr <= 0:
            return None

        dc_upper = self._get_dc_value(series.donchian_upper)
        dc_lower = self._get_dc_value(series.donchian_lower)

        if dc_upper is None or dc_lower is None:
            return None

        is_squeezed = bb_width < 0.025

        broke_upper = price > dc_upper and prev_price <= dc_upper
        broke_lower = price < dc_lower and prev_price >= dc_lower

        if not (broke_upper or broke_lower):
            return None

        if vol_ratio < params.vol_ratio_breakout:
            return None

        if adx < 18:
            return None

        if broke_upper and rsi > 80:
            return None
        if broke_lower and rsi < 20:
            return None

        # HTF filter: don't break out against a strong higher timeframe trend
        if broke_upper and htf_bias < -0.5:
            return None
        if broke_lower and htf_bias > 0.5:
            return None

        # ── Strength calculation ────────────────────────────────────────────

        strength = 0.0

        if is_squeezed:
            squeeze_score = min(0.3, (0.025 - bb_width) * 15)
            strength += max(0, squeeze_score)

        vol_score = min(0.25, (vol_ratio - 1.0) * 0.2)
        strength += max(0, vol_score)

        if broke_upper:
            breakout_dist = (price - dc_upper) / raw_atr
        else:
            breakout_dist = (dc_lower - price) / raw_atr
        dist_score = min(0.25, breakout_dist * 0.15)
        strength += max(0, dist_score)

        adx_score = min(0.2, (adx - 15) / 100)
        strength += max(0, adx_score)

        # HTF agreement bonus
        if (broke_upper and htf_bias > 0.3) or (broke_lower and htf_bias < -0.3):
            strength += 0.1

        strength = round(min(1.0, strength), 4)

        # ── Stops ───────────────────────────────────────────────────────────

        if broke_upper:
            direction = Direction.LONG
            stop_loss = price - params.atr_stop_breakout * raw_atr
            take_profit = price + params.atr_target_breakout * raw_atr
        else:
            direction = Direction.SHORT
            stop_loss = price + params.atr_stop_breakout * raw_atr
            take_profit = price - params.atr_target_breakout * raw_atr

        confidence = min(1.0, strength * (1.0 if is_squeezed else 0.7))

        return Signal(
            instrument=series.instrument,
            source=SignalSource.BREAKOUT,
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
                f"Breakout {'up' if broke_upper else 'down'}: "
                f"DC_pct={dc_pct:.2f} BB_width={bb_width:.4f} "
                f"vol_ratio={vol_ratio:.1f} ADX={adx:.0f} "
                f"HTF={htf_bias:+.2f} {'SQUEEZED ' if is_squeezed else ''}"
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
    def _get_dc_value(arr: Optional[list[float]]) -> Optional[float]:
        if not arr:
            return None
        for v in reversed(arr[:-1]):
            if v is not None and not np.isnan(v):
                return float(v)
        return None
