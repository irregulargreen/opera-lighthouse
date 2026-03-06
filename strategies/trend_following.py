"""
Trend following strategy — the most documented persistent edge in markets.

Exploits: disposition effect, anchoring, herding, institutional momentum flows.

Entry: multiple timeframe MA alignment + ADX confirmation + MACD momentum
Exit: ATR-based trailing stop (lets winners run, cuts losers short)

Typical stats: ~42% win rate, 2:1+ reward:risk ratio.
Profitable because winners are much larger than losers.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from core.instrument_params import get_params
from core.models import Direction, PriceSeries, Regime, Signal, SignalSource
from strategies.base import Strategy


class TrendFollowing(Strategy):

    @property
    def name(self) -> str:
        return "trend_following"

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
        adx = features.get("adx", 0.0)
        ma_align = features.get("ma_alignment", 0.0)
        rsi = features.get("rsi", 50.0)
        macd_hist = features.get("macd_histogram", 0.0)
        ret_20 = features.get("ret_20", 0.0)
        htf_bias = features.get("htf_bias", 0.0)

        if adx < params.adx_trend_min:
            return None

        if abs(ma_align) < params.ma_align_min:
            return None

        is_long = ma_align > params.ma_align_min
        is_short = ma_align < -params.ma_align_min

        if not (is_long or is_short):
            return None

        if is_long and macd_hist < 0:
            return None
        if is_short and macd_hist > 0:
            return None

        if is_long and rsi > 75:
            return None
        if is_short and rsi < 25:
            return None

        if is_long and ret_20 < 0:
            return None
        if is_short and ret_20 > 0:
            return None

        # HTF filter: don't trade against the higher timeframe trend
        if is_long and htf_bias < -0.3:
            return None
        if is_short and htf_bias > 0.3:
            return None

        # ── Strength calculation ────────────────────────────────────────────

        strength = 0.0

        adx_score = min(0.4, max(0.0, (adx - 20) / 75))
        strength += adx_score

        align_score = min(0.3, abs(ma_align) * 0.3)
        strength += align_score

        macd_score = min(0.2, abs(macd_hist) * 0.15)
        strength += macd_score

        if (is_long and ret_20 > 0) or (is_short and ret_20 < 0):
            strength += 0.1

        # HTF agreement bonus
        if (is_long and htf_bias > 0.3) or (is_short and htf_bias < -0.3):
            strength += 0.1

        strength = round(min(1.0, strength), 4)

        # ── Stop loss and take profit ───────────────────────────────────────

        raw_atr = self._get_raw_atr(series)
        if raw_atr <= 0:
            return None

        if is_long:
            direction = Direction.LONG
            stop_loss = price - params.atr_stop_trend * raw_atr
            take_profit = price + params.atr_target_trend * raw_atr
        else:
            direction = Direction.SHORT
            stop_loss = price + params.atr_stop_trend * raw_atr
            take_profit = price - params.atr_target_trend * raw_atr

        confidence = min(1.0, (adx / 50) * abs(ma_align))

        return Signal(
            instrument=series.instrument,
            source=SignalSource.TREND,
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
                f"Trend {'up' if is_long else 'down'}: "
                f"ADX={adx:.0f} MA_align={ma_align:+.2f} "
                f"MACD_hist={macd_hist:+.4f} RSI={rsi:.0f} "
                f"HTF={htf_bias:+.2f}"
            ),
        )

    @staticmethod
    def _get_raw_atr(series: PriceSeries) -> float:
        if series.atr_14:
            valid = [v for v in series.atr_14[-5:] if v is not None and not np.isnan(v)]
            if valid:
                return float(valid[-1])
        return 0.0
