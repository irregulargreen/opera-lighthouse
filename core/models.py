"""
models.py — shared Pydantic models used throughout the bot.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class MarketOutcome(str, Enum):
    YES = "YES"
    NO = "NO"


class MarketStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"


class Market(BaseModel):
    """Normalised market object from Gamma API."""
    condition_id: str
    question_id: str
    token_id_yes: str               # CLOB token ID for YES outcome
    token_id_no: str                # CLOB token ID for NO outcome
    question: str
    description: Optional[str] = None
    resolution_criteria: Optional[str] = None
    category: Optional[str] = None
    tags: list[str] = Field(default_factory=list)

    # Prices (0.0 – 1.0, i.e. cents / 100)
    yes_price: float = 0.5
    no_price: float = 0.5

    # Liquidity / volume
    volume_24h: float = 0.0
    volume_total: float = 0.0
    liquidity: float = 0.0          # USDC in order book

    # Timing
    created_at: Optional[datetime] = None
    close_time: Optional[datetime] = None

    # Derived
    is_new: bool = False            # Created in last N minutes
    spread: float = 0.0             # ask - bid


class OrderBook(BaseModel):
    token_id: str
    bids: list[tuple[float, float]] = Field(default_factory=list)  # (price, size)
    asks: list[tuple[float, float]] = Field(default_factory=list)
    best_bid: float = 0.0
    best_ask: float = 1.0
    mid: float = 0.5
    spread: float = 1.0
    depth_bid_usdc: float = 0.0     # Total USDC on bid side
    depth_ask_usdc: float = 0.0


class NewsItem(BaseModel):
    title: str
    description: Optional[str] = None
    url: str
    published_at: Optional[datetime] = None
    source: str = ""
    relevance_score: float = 0.0    # 0-1, set by enricher


class EnrichedMarket(BaseModel):
    market: Market
    news_items: list[NewsItem] = Field(default_factory=list)
    news_summary: str = ""          # Condensed context for LLM
    related_markets: list[str] = Field(default_factory=list)  # Correlated market questions


class LLMEstimate(BaseModel):
    provider: str                   # "openai", "anthropic", "ollama"
    model: str
    probability: float              # 0.0 – 1.0
    confidence: float               # 0.0 – 1.0 (model's self-assessed certainty)
    reasoning: str = ""
    base_rate: Optional[float] = None
    raw_response: str = ""


class EnsembleEstimate(BaseModel):
    """Combined estimate from multiple LLMs."""
    raw_estimates: list[LLMEstimate] = Field(default_factory=list)
    ensemble_probability: float     # Weighted average before shrinkage
    calibrated_probability: float   # After shrinkage toward 0.5
    final_probability: float        # After crowd-weighting
    confidence: float               # Average confidence
    consensus: float = 0.0          # Std dev of estimates (low = high consensus)
    reasoning_summary: str = ""


class TradeSignal(BaseModel):
    """A trade opportunity identified by the scanner."""
    market: Market
    ensemble: EnsembleEstimate
    side: MarketOutcome             # Which outcome to buy
    token_id: str                   # Which token to buy
    market_price: float             # Current price of that token
    estimated_probability: float    # Our estimate
    raw_edge: float                 # estimated_prob - market_price
    edge_after_fees: float          # Accounting for taker fee
    kelly_size_usdc: float          # Optimal position size
    capped_size_usdc: float         # After max-size cap
    expected_value: float           # edge × size
    rationale: str = ""
    # Attached at pipeline time for meta-learner access (not persisted)
    enriched_context: Optional["EnrichedMarket"] = None

    model_config = {"arbitrary_types_allowed": True}


class TradeResult(BaseModel):
    """Outcome of an executed or simulated trade."""
    signal: TradeSignal
    order_id: Optional[str] = None
    executed: bool = False
    dry_run: bool = True
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ResolvedTrade(BaseModel):
    """A trade that has resolved — used for calibration."""
    trade: TradeResult
    resolved_outcome: MarketOutcome  # What actually happened
    pnl_usdc: float
    our_probability: float           # What we estimated
    market_probability: float        # What market priced
    brier_score: float               # (outcome - probability)^2
    resolved_at: datetime = Field(default_factory=datetime.utcnow)