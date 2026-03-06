"""
edge_calculator.py — converts LLM probability estimates into concrete trade signals.

Handles:
  - Edge calculation (accounting for fees, spread, slippage)
  - Kelly criterion position sizing for binary markets
  - Liquidity/slippage checks
  - Signal filtering and ranking
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from core.config import config
from core.models import (
    EnrichedMarket,
    EnsembleEstimate,
    Market,
    MarketOutcome,
    TradeSignal,
)


class EdgeCalculator:
    """
    Given an enriched market and its ensemble LLM estimate, determines:
      - Whether a tradeable edge exists
      - Which side to bet (YES or NO)
      - How much to bet (Kelly-sized, capped)
    """

    def evaluate(
        self,
        enriched: EnrichedMarket,
        ensemble: EnsembleEstimate,
    ) -> Optional[TradeSignal]:
        """
        Main entry point. Returns a TradeSignal if an edge exists, else None.
        """
        market = enriched.market
        final_prob = ensemble.final_probability

        # ── Pre-flight checks ─────────────────────────────────────────────────

        # Skip if LLM confidence is too low
        if ensemble.confidence < config.bot.min_llm_confidence:
            logger.debug(
                f"[Edge] SKIP low confidence ({ensemble.confidence:.2f}): {market.question[:60]}"
            )
            return None

        # Skip if model consensus is poor (high std dev = models disagree)
        if ensemble.consensus > 0.20:
            logger.debug(
                f"[Edge] SKIP poor consensus ({ensemble.consensus:.2f}): {market.question[:60]}"
            )
            return None

        # Skip if market is closing very soon (not enough time for price to converge)
        if market.close_time:
            from datetime import datetime, timezone
            remaining = (market.close_time - datetime.now(timezone.utc)).total_seconds()
            if remaining < 3600:  # Less than 1 hour to close
                logger.debug(f"[Edge] SKIP closing soon: {market.question[:60]}")
                return None

        # Skip if liquidity is too thin
        if market.liquidity < config.bot.min_liquidity_usdc:
            logger.debug(
                f"[Edge] SKIP illiquid (${market.liquidity:.0f}): {market.question[:60]}"
            )
            return None

        # ── Determine which side has the edge ─────────────────────────────────

        # YES edge: we think YES is more likely than the market price
        yes_edge = final_prob - market.yes_price
        # NO edge: we think NO is more likely (i.e. YES is overpriced)
        no_edge = (1 - final_prob) - market.no_price

        if yes_edge >= no_edge and yes_edge > 0:
            side = MarketOutcome.YES
            token_id = market.token_id_yes
            market_price = market.yes_price
            our_prob = final_prob
        elif no_edge > yes_edge and no_edge > 0:
            side = MarketOutcome.NO
            token_id = market.token_id_no
            market_price = market.no_price
            our_prob = 1.0 - final_prob
        else:
            logger.debug(f"[Edge] No edge found: {market.question[:60]}")
            return None

        raw_edge = our_prob - market_price

        # ── Fee adjustment ────────────────────────────────────────────────────
        # Taker fee is paid on the amount wagered
        # True edge = (p_win × payout) - (1 × cost) - fee
        # Payout per share = 1/market_price; cost = 1; fee = taker_fee × size
        edge_after_fees = raw_edge - config.bot.taker_fee

        if edge_after_fees < config.bot.min_edge:
            logger.debug(
                f"[Edge] SKIP edge {edge_after_fees:.3f} < min {config.bot.min_edge}: "
                f"{market.question[:60]}"
            )
            return None

        # ── Slippage estimate ─────────────────────────────────────────────────
        # For small orders on decent liquidity, slippage is roughly:
        # slippage ≈ trade_size / depth_of_book
        # We penalise the edge if slippage is material
        estimated_slippage = self._estimate_slippage(
            config.bot.max_trade_size_usdc, market.liquidity
        )
        edge_after_fees -= estimated_slippage

        if edge_after_fees < config.bot.min_edge:
            logger.debug(f"[Edge] SKIP after slippage: {market.question[:60]}")
            return None

        # ── Kelly position sizing ─────────────────────────────────────────────
        kelly_size = self._kelly_size(
            our_prob=our_prob,
            market_price=market_price,
            total_capital=config.bot.total_capital_usdc,
        )

        # Apply fractional Kelly and hard cap
        kelly_size = kelly_size * config.bot.kelly_fraction
        capped_size = min(kelly_size, config.bot.max_trade_size_usdc)

        # Don't trade if Kelly says size is negligible (< $1)
        if capped_size < 1.0:
            logger.debug(f"[Edge] SKIP Kelly size too small: {market.question[:60]}")
            return None

        expected_value = edge_after_fees * capped_size

        rationale = (
            f"Our P({side.value}): {our_prob:.1%} vs market: {market_price:.1%} | "
            f"Edge after fees: {edge_after_fees:.1%} | "
            f"Confidence: {ensemble.confidence:.0%} | "
            f"Consensus: ±{ensemble.consensus:.2f} | "
            f"Size: ${capped_size:.2f} | EV: ${expected_value:.2f}\n"
            f"LLM reasoning: {ensemble.reasoning_summary[:300]}"
        )

        signal = TradeSignal(
            market=market,
            ensemble=ensemble,
            side=side,
            token_id=token_id,
            market_price=market_price,
            estimated_probability=our_prob,
            raw_edge=round(raw_edge, 4),
            edge_after_fees=round(edge_after_fees, 4),
            kelly_size_usdc=round(kelly_size, 2),
            capped_size_usdc=round(capped_size, 2),
            expected_value=round(expected_value, 4),
            rationale=rationale,
        )

        logger.info(
            f"[Edge] ✓ SIGNAL {side.value} {market.question[:55]!r} | "
            f"market={market_price:.2%} ours={our_prob:.2%} edge={edge_after_fees:.2%} "
            f"size=${capped_size:.2f}"
        )
        return signal

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _kelly_size(our_prob: float, market_price: float, total_capital: float) -> float:
        """
        Kelly Criterion for binary prediction markets (bet-to-win format).

        In a prediction market:
          - You buy shares at `market_price` (cost per share)
          - Each share pays out $1.00 if you win
          - Net odds b = (1 - market_price) / market_price  [profit per dollar risked]

        Kelly formula: f* = (b × p - q) / b
          where p = our probability, q = 1 - p, b = net odds

        This gives the fraction of capital to wager.
        """
        if market_price <= 0 or market_price >= 1:
            return 0.0

        b = (1.0 - market_price) / market_price  # Net odds
        p = our_prob
        q = 1.0 - p

        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0.0, kelly_fraction)

        return kelly_fraction * total_capital

    @staticmethod
    def _estimate_slippage(trade_size_usdc: float, book_liquidity_usdc: float) -> float:
        """
        Simple linear slippage model: slippage ≈ trade_size / book_depth × 0.5
        Returns slippage as a fraction (e.g. 0.005 = 0.5%)
        """
        if book_liquidity_usdc <= 0:
            return 0.05  # 5% penalty for unknown liquidity
        slippage = (trade_size_usdc / book_liquidity_usdc) * 0.5
        return min(slippage, 0.05)  # Cap at 5%

    @staticmethod
    def rank_signals(signals: list[TradeSignal]) -> list[TradeSignal]:
        """
        Rank signals by expected value, with bonuses for:
          - New markets (first-mover advantage)
          - High consensus (models agree)
          - High confidence
        """
        def score(s: TradeSignal) -> float:
            base = s.expected_value
            new_market_bonus = 1.5 if s.market.is_new else 1.0
            consensus_factor = 1.0 + (0.20 - s.ensemble.consensus)  # Less disagreement = higher score
            confidence_factor = s.ensemble.confidence
            return base * new_market_bonus * consensus_factor * confidence_factor

        return sorted(signals, key=score, reverse=True)