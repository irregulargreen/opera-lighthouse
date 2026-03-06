"""Per-instrument trading parameters.

Different instruments have different volatility profiles, spread costs,
and trend characteristics. These params tune strategy behavior per pair
rather than using one-size-fits-all thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentParams:
    atr_stop_trend: float = 2.5
    atr_target_trend: float = 4.0
    atr_stop_breakout: float = 2.5
    atr_target_breakout: float = 5.0
    atr_stop_reversion: float = 2.0
    atr_target_reversion: float = 2.0
    adx_trend_min: float = 25.0
    ma_align_min: float = 0.5
    adx_reversion_max: float = 22.0
    vol_ratio_breakout: float = 1.3
    enable_reversion: bool = True


_PARAMS: dict[str, InstrumentParams] = {
    "EUR_USD": InstrumentParams(),

    "GBP_USD": InstrumentParams(
        atr_stop_trend=2.5,
        atr_target_trend=4.5,
        atr_stop_breakout=2.5,
        atr_target_breakout=5.0,
    ),

    "AUD_USD": InstrumentParams(
        adx_trend_min=23.0,
        atr_target_trend=4.5,
        atr_target_breakout=5.5,
    ),

    "USD_CAD": InstrumentParams(
        atr_stop_trend=3.0,
        atr_target_trend=4.5,
        atr_stop_breakout=3.0,
        atr_target_breakout=5.0,
        adx_trend_min=28.0,
        enable_reversion=False,
    ),

    "EUR_GBP": InstrumentParams(
        atr_stop_trend=2.0,
        atr_target_trend=3.0,
        atr_stop_breakout=2.0,
        atr_target_breakout=4.0,
        adx_trend_min=22.0,
        adx_reversion_max=20.0,
    ),

    "XAU_USD": InstrumentParams(
        atr_stop_trend=3.0,
        atr_target_trend=5.0,
        atr_stop_breakout=3.0,
        atr_target_breakout=6.0,
        adx_trend_min=22.0,
        enable_reversion=False,
    ),

    "USD_CHF": InstrumentParams(
        adx_trend_min=25.0,
    ),
}

_DEFAULT = InstrumentParams()


def get_params(instrument: str) -> InstrumentParams:
    return _PARAMS.get(instrument, _DEFAULT)
