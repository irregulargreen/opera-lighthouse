"""
executor.py — wraps py-clob-client to place, track, and cancel orders.

Supports:
  - Dry run mode (logs trades without executing)
  - Limit orders (GTC) for better pricing on less liquid markets
  - Market orders (FOK) for highly liquid markets with urgent edge
  - Position tracking across the session
  - Auto-cancel stale open orders
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from core.config import config
from core.models import MarketOutcome, TradeResult, TradeSignal


class Executor:
    """
    Executes trades via Polymarket CLOB API.
    Uses py-clob-client under the hood.
    """

    def __init__(self):
        self._client = None
        self._open_positions: dict[str, TradeResult] = {}  # order_id → result
        self._total_invested = 0.0

    def initialise(self):
        """
        Set up the py-clob-client. Call this once at startup.
        Skipped in dry-run mode.
        """
        if config.bot.dry_run:
            logger.info("[Executor] DRY RUN mode — no real orders will be placed")
            return

        try:
            from py_clob_client.client import ClobClient

            self._client = ClobClient(
                config.polymarket.clob_host,
                key=config.polymarket.private_key,
                chain_id=config.polymarket.chain_id,
                signature_type=config.polymarket.signature_type,
                funder=config.polymarket.funder_address,
            )
            # Derive and set API credentials (signed from private key)
            creds = self._client.create_or_derive_api_creds()
            self._client.set_api_creds(creds)

            # Verify connection
            ok = self._client.get_ok()
            logger.info(f"[Executor] Connected to Polymarket CLOB: {ok}")

        except ImportError:
            logger.error("[Executor] py-clob-client not installed. Run: pip install py-clob-client")
            raise
        except Exception as e:
            logger.error(f"[Executor] Failed to initialise CLOB client: {e}")
            raise

    # ─── Public interface ─────────────────────────────────────────────────────

    def execute(self, signal: TradeSignal) -> TradeResult:
        """
        Execute a trade signal. Returns a TradeResult regardless of dry/live mode.
        """
        if len(self._open_positions) >= config.bot.max_open_positions:
            logger.warning("[Executor] Max open positions reached — skipping")
            return TradeResult(
                signal=signal,
                executed=False,
                dry_run=config.bot.dry_run,
                error="Max open positions reached",
            )

        if config.bot.dry_run:
            return self._dry_run_trade(signal)
        else:
            return self._live_trade(signal)

    def get_open_positions(self) -> dict[str, TradeResult]:
        return self._open_positions.copy()

    def get_total_invested(self) -> float:
        return self._total_invested

    def cancel_all_orders(self):
        """Cancel all open orders. Call on shutdown."""
        if config.bot.dry_run or not self._client:
            return
        try:
            result = self._client.cancel_all()
            logger.info(f"[Executor] Cancelled all orders: {result}")
        except Exception as e:
            logger.error(f"[Executor] Failed to cancel orders: {e}")

    def get_balance(self) -> Optional[float]:
        """Get current USDC balance from the CLOB API."""
        if config.bot.dry_run or not self._client:
            return config.bot.total_capital_usdc - self._total_invested
        try:
            # py-clob-client doesn't have a direct balance method
            # Balance is tracked via the Data API
            return None
        except Exception:
            return None

    # ─── Execution modes ──────────────────────────────────────────────────────

    def _dry_run_trade(self, signal: TradeSignal) -> TradeResult:
        """Simulate a trade without placing an order."""
        fake_order_id = f"DRY-{signal.market.question_id[:8]}-{int(datetime.now().timestamp())}"

        self._total_invested += signal.capped_size_usdc
        result = TradeResult(
            signal=signal,
            order_id=fake_order_id,
            executed=True,
            dry_run=True,
        )
        self._open_positions[fake_order_id] = result

        logger.info(
            f"[Executor] 🧪 DRY RUN | {signal.side.value} {signal.market.question[:55]!r}\n"
            f"  Price: {signal.market_price:.3f} | Size: ${signal.capped_size_usdc:.2f} | "
            f"Order ID: {fake_order_id}"
        )
        return result

    def _live_trade(self, signal: TradeSignal) -> TradeResult:
        """Place a real order via py-clob-client."""
        try:
            # Choose order type:
            # - High liquidity + urgent edge: FOK market order (guaranteed fill or cancel)
            # - Normal: GTC limit order slightly above mid (better price, may not fill immediately)
            use_market_order = signal.market.liquidity > 50_000 and signal.edge_after_fees > 0.08

            if use_market_order:
                result = self._place_market_order(signal)
            else:
                result = self._place_limit_order(signal)

            if result.executed:
                self._open_positions[result.order_id] = result
                self._total_invested += signal.capped_size_usdc

            return result

        except Exception as e:
            logger.error(f"[Executor] Trade failed: {e}")
            return TradeResult(
                signal=signal,
                executed=False,
                dry_run=False,
                error=str(e),
            )

    def _place_market_order(self, signal: TradeSignal) -> TradeResult:
        """Place a FOK (fill-or-kill) market order."""
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        order_args = MarketOrderArgs(
            token_id=signal.token_id,
            amount=signal.capped_size_usdc,
            side=BUY,
            order_type=OrderType.FOK,
        )
        signed_order = self._client.create_market_order(order_args)
        resp = self._client.post_order(signed_order, OrderType.FOK)

        order_id = resp.get("orderID") or resp.get("id", "")
        success = resp.get("success", False) or bool(order_id)

        logger.info(
            f"[Executor] ⚡ MARKET ORDER | {signal.side.value} {signal.market.question[:50]!r} | "
            f"${signal.capped_size_usdc:.2f} | ID: {order_id} | success={success}"
        )

        return TradeResult(
            signal=signal,
            order_id=order_id,
            executed=success,
            dry_run=False,
            error=None if success else str(resp),
        )

    def _place_limit_order(self, signal: TradeSignal) -> TradeResult:
        """
        Place a GTC limit order at a price slightly better than current market.
        For YES: bid slightly above best bid (more likely to fill)
        For NO: same logic on NO token
        """
        from py_clob_client.clob_types import LimitOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        # Price: split the spread — go 40% of the way from mid toward our estimate
        # This gives us a better price than market while still being competitive
        spread = signal.market.spread if hasattr(signal.market, "spread") else 0.02
        limit_price = round(signal.market_price + spread * 0.4, 4)
        limit_price = min(limit_price, signal.estimated_probability - 0.01)  # Don't overpay

        # Convert USDC size to number of shares
        # shares = usdc / price_per_share
        shares = signal.capped_size_usdc / limit_price

        order_args = LimitOrderArgs(
            token_id=signal.token_id,
            price=limit_price,
            size=round(shares, 2),
            side=BUY,
        )
        signed_order = self._client.create_limit_order(order_args)
        resp = self._client.post_order(signed_order, OrderType.GTC)

        order_id = resp.get("orderID") or resp.get("id", "")
        success = resp.get("success", False) or bool(order_id)

        logger.info(
            f"[Executor] 📋 LIMIT ORDER | {signal.side.value} {signal.market.question[:50]!r} | "
            f"${signal.capped_size_usdc:.2f} @ {limit_price:.4f} | ID: {order_id}"
        )

        return TradeResult(
            signal=signal,
            order_id=order_id,
            executed=success,
            dry_run=False,
            error=None if success else str(resp),
        )