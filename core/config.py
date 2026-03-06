"""
config.py — centralised settings, loaded once at startup.
All other modules import from here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def _require(key: str) -> str:
    v = os.getenv(key)
    if not v:
        raise EnvironmentError(f"Required env var '{key}' is not set. Check your .env file.")
    return v


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


@dataclass
class PolymarketConfig:
    private_key: str = field(default_factory=lambda: _require("POLYMARKET_PRIVATE_KEY"))
    funder_address: str = field(default_factory=lambda: _require("POLYMARKET_FUNDER_ADDRESS"))
    signature_type: int = field(default_factory=lambda: _int("POLYMARKET_SIGNATURE_TYPE", 0))
    clob_host: str = "https://clob.polymarket.com"
    gamma_host: str = "https://gamma-api.polymarket.com"
    data_host: str = "https://data-api.polymarket.com"
    chain_id: int = 137  # Polygon mainnet


@dataclass
class LLMConfig:
    openai_api_key: str = field(default_factory=lambda: _optional("OPENAI_API_KEY"))
    anthropic_api_key: str = field(default_factory=lambda: _optional("ANTHROPIC_API_KEY"))
    ollama_base_url: str = field(default_factory=lambda: _optional("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _optional("OLLAMA_MODEL", "llama3.1:8b"))

    # Which providers are enabled (auto-detected from key presence)
    @property
    def openai_enabled(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def anthropic_enabled(self) -> bool:
        return bool(self.anthropic_api_key)

    # Ollama is assumed available if URL is set
    @property
    def ollama_enabled(self) -> bool:
        return bool(self.ollama_base_url)


@dataclass
class NewsConfig:
    newsapi_key: str = field(default_factory=lambda: _optional("NEWSAPI_KEY"))

    @property
    def newsapi_enabled(self) -> bool:
        return bool(self.newsapi_key)


@dataclass
class BotConfig:
    min_edge: float = field(default_factory=lambda: _float("MIN_EDGE", 0.05))
    max_trade_size_usdc: float = field(default_factory=lambda: _float("MAX_TRADE_SIZE_USDC", 50.0))
    kelly_fraction: float = field(default_factory=lambda: _float("KELLY_FRACTION", 0.25))
    total_capital_usdc: float = field(default_factory=lambda: _float("TOTAL_CAPITAL_USDC", 500.0))
    min_liquidity_usdc: float = field(default_factory=lambda: _float("MIN_LIQUIDITY_USDC", 1000.0))
    max_open_positions: int = field(default_factory=lambda: _int("MAX_OPEN_POSITIONS", 10))
    scan_interval_seconds: int = field(default_factory=lambda: _int("SCAN_INTERVAL_SECONDS", 120))
    full_sweep_interval_seconds: int = field(default_factory=lambda: _int("FULL_SWEEP_INTERVAL_SECONDS", 900))
    min_llm_confidence: float = field(default_factory=lambda: _float("MIN_LLM_CONFIDENCE", 0.60))
    dry_run: bool = field(default_factory=lambda: _bool("DRY_RUN", True))

    # Taker fee on Polymarket (check current docs — was ~2% on crypto markets post Feb 2026)
    taker_fee: float = 0.02
    # Maker fee (negative = rebate for limit orders that add liquidity)
    maker_fee: float = -0.001

    # Calibration: shrink LLM estimates toward 0.5 to correct overconfidence
    llm_shrinkage: float = 0.15

    # How much to weight crowd price vs LLM estimate in final estimate
    # Based on Halawi et al NeurIPS 2024: 0.7 crowd / 0.3 LLM is a strong default
    crowd_weight: float = 0.30
    llm_weight: float = 0.70


@dataclass
class Config:
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    bot: BotConfig = field(default_factory=BotConfig)


# Singleton — import this everywhere
config = Config()