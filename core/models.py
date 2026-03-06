"""Shared data types used across the system."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class Regime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class SignalSource(str, Enum):
    TREND = "trend_following"
    REVERSION = "mean_reversion"
    BREAKOUT = "breakout"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


# ── Market Data ───────────────────────────────────────────────────────────────


class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


class PriceSeries(BaseModel):
    """A time series of candles with computed indicators attached."""
    instrument: str
    timeframe: str
    candles: list[Candle] = Field(default_factory=list)

    # Technical indicators — populated by FeatureEngine
    sma_20: Optional[list[float]] = None
    sma_50: Optional[list[float]] = None
    sma_200: Optional[list[float]] = None
    ema_12: Optional[list[float]] = None
    ema_26: Optional[list[float]] = None
    rsi_14: Optional[list[float]] = None
    macd_line: Optional[list[float]] = None
    macd_signal: Optional[list[float]] = None
    macd_histogram: Optional[list[float]] = None
    bb_upper: Optional[list[float]] = None
    bb_middle: Optional[list[float]] = None
    bb_lower: Optional[list[float]] = None
    atr_14: Optional[list[float]] = None
    adx_14: Optional[list[float]] = None
    donchian_upper: Optional[list[float]] = None
    donchian_lower: Optional[list[float]] = None
    returns: Optional[list[float]] = None
    volatility_20: Optional[list[float]] = None

    @property
    def closes(self) -> np.ndarray:
        return np.array([c.close for c in self.candles])

    @property
    def highs(self) -> np.ndarray:
        return np.array([c.high for c in self.candles])

    @property
    def lows(self) -> np.ndarray:
        return np.array([c.low for c in self.candles])

    @property
    def volumes(self) -> np.ndarray:
        return np.array([c.volume for c in self.candles])

    @property
    def latest(self) -> Optional[Candle]:
        return self.candles[-1] if self.candles else None

    model_config = {"arbitrary_types_allowed": True}


# ── Signals & Trades ─────────────────────────────────────────────────────────


class Signal(BaseModel):
    """A trading signal from any strategy."""
    instrument: str
    source: SignalSource
    direction: Direction
    strength: float                    # 0.0–1.0, how strong the signal is
    confidence: float = 0.5            # 0.0–1.0, strategy's confidence
    entry_price: float = 0.0           # Suggested entry
    stop_loss: float = 0.0             # Hard stop loss
    take_profit: float = 0.0           # Target price
    atr: float = 0.0                   # Current ATR (for position sizing)
    regime: Regime = Regime.UNKNOWN
    features: dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Position(BaseModel):
    """An open or closed position."""
    trade_id: str
    instrument: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    size_units: float                  # Number of units (not lots)
    risk_usdc: float                   # Dollar amount at risk
    signal: Signal

    status: TradeStatus = TradeStatus.OPEN
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pips: Optional[float] = None
    exit_reason: str = ""

    opened_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None

    # ML features at time of entry (for meta-learning)
    entry_features: dict[str, float] = Field(default_factory=dict)


class PerformanceSnapshot(BaseModel):
    """Rolling performance metrics for a strategy or the whole system."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    expectancy: float = 0.0           # avg_win * win_rate - avg_loss * (1 - win_rate)

    @property
    def edge_per_trade(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades
