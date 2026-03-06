"""Centralised settings loaded from .env at startup."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def _str(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


def _list(key: str, default: str = "") -> list[str]:
    raw = os.getenv(key, default)
    return [s.strip() for s in raw.split(",") if s.strip()]


@dataclass
class OANDAConfig:
    api_key: str = field(default_factory=lambda: _str("OANDA_API_KEY"))
    account_id: str = field(default_factory=lambda: _str("OANDA_ACCOUNT_ID"))
    environment: str = field(default_factory=lambda: _str("OANDA_ENVIRONMENT", "practice"))

    @property
    def base_url(self) -> str:
        if self.environment == "live":
            return "https://api-fxtrade.oanda.com"
        return "https://api-fxpractice.oanda.com"

    @property
    def stream_url(self) -> str:
        if self.environment == "live":
            return "https://stream-fxtrade.oanda.com"
        return "https://stream-fxpractice.oanda.com"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.account_id)


@dataclass
class TradingConfig:
    dry_run: bool = field(default_factory=lambda: _bool("DRY_RUN", True))
    instruments: list[str] = field(default_factory=lambda: _list("INSTRUMENTS", "EUR_USD"))
    total_capital: float = field(default_factory=lambda: _float("TOTAL_CAPITAL", 10000.0))
    max_risk_per_trade: float = field(default_factory=lambda: _float("MAX_RISK_PER_TRADE", 0.02))
    max_open_positions: int = field(default_factory=lambda: _int("MAX_OPEN_POSITIONS", 6))
    max_correlation_exposure: int = field(default_factory=lambda: _int("MAX_CORRELATION_EXPOSURE", 3))
    kelly_fraction: float = field(default_factory=lambda: _float("KELLY_FRACTION", 0.25))
    min_signal_strength: float = field(default_factory=lambda: _float("MIN_SIGNAL_STRENGTH", 0.45))


@dataclass
class StrategyConfig:
    trend_timeframes: list[str] = field(default_factory=lambda: _list("TREND_TIMEFRAMES", "H1,H4,D"))
    reversion_timeframe: str = field(default_factory=lambda: _str("REVERSION_TIMEFRAME", "M15"))
    lookback_bars: int = field(default_factory=lambda: _int("LOOKBACK_BARS", 500))


@dataclass
class MLConfig:
    regime_retrain_hours: int = field(default_factory=lambda: _int("REGIME_RETRAIN_HOURS", 24))
    min_trades_for_ml: int = field(default_factory=lambda: _int("MIN_TRADES_FOR_ML", 50))


@dataclass
class Config:
    oanda: OANDAConfig = field(default_factory=OANDAConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    ml: MLConfig = field(default_factory=MLConfig)


config = Config()
