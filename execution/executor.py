"""
Trade executor — places, monitors, and closes trades via OANDA.

Handles:
  - Dry run (paper trading) and live execution
  - Position tracking with SL/TP
  - Trade closure detection and P&L calculation
  - Position recovery after restart (reloads open trades from SQLite)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from core.config import config
from core.models import Direction, Position, Signal, SignalSource, Regime, TradeStatus
from data.market_data import OANDAClient
from data.store import TradeStore


# Pip size per instrument type
def _pip_size(instrument: str) -> float:
    if "JPY" in instrument:
        return 0.01
    if instrument.startswith("XAU"):
        return 0.1
    if instrument.startswith("XAG"):
        return 0.01
    return 0.0001


class Executor:

    def __init__(self, store: TradeStore, oanda: OANDAClient):
        self._store = store
        self._oanda = oanda
        self._open_positions: dict[str, Position] = {}
        self._recover_open_positions()

    def _recover_open_positions(self):
        """Reload open positions from SQLite after process restart."""
        rows = self._store.get_open_trades()
        if not rows:
            return

        recovered = 0
        for row in rows:
            try:
                features = {}
                feat_str = row.get("entry_features")
                if feat_str:
                    try:
                        features = json.loads(feat_str)
                    except (json.JSONDecodeError, TypeError):
                        pass

                source_val = row.get("strategy", "trend_following")
                try:
                    source = SignalSource(source_val)
                except ValueError:
                    source = SignalSource.TREND

                regime_val = row.get("regime", "unknown")
                try:
                    regime = Regime(regime_val)
                except ValueError:
                    regime = Regime.UNKNOWN

                signal = Signal(
                    instrument=row.get("instrument", ""),
                    source=source,
                    direction=Direction(row.get("direction", "long")),
                    strength=row.get("signal_strength", 0.5),
                    confidence=row.get("signal_confidence", 0.5),
                    entry_price=row.get("entry_price", 0.0),
                    stop_loss=row.get("stop_loss", 0.0),
                    take_profit=row.get("take_profit", 0.0),
                    regime=regime,
                    features=features,
                )

                position = Position(
                    trade_id=row["trade_id"],
                    instrument=row.get("instrument", ""),
                    direction=Direction(row.get("direction", "long")),
                    entry_price=row.get("entry_price", 0.0),
                    stop_loss=row.get("stop_loss", 0.0),
                    take_profit=row.get("take_profit", 0.0),
                    size_units=row.get("size_units", 0.0),
                    risk_usdc=row.get("risk_amount", 0.0),
                    signal=signal,
                    entry_features=features,
                    opened_at=datetime.fromisoformat(row["opened_at"]) if row.get("opened_at") else datetime.now(timezone.utc),
                )

                self._open_positions[position.trade_id] = position
                recovered += 1
            except Exception as e:
                logger.warning(f"[Executor] Failed to recover trade {row.get('trade_id')}: {e}")

        if recovered:
            logger.info(f"[Executor] Recovered {recovered} open position(s) from database")

    async def execute(self, signal: Signal, units: int, risk_usdc: float) -> Optional[Position]:
        trade_id = f"{'DRY' if config.trading.dry_run else 'LIVE'}-{uuid.uuid4().hex[:12]}"

        position = Position(
            trade_id=trade_id,
            instrument=signal.instrument,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            size_units=float(units),
            risk_usdc=risk_usdc,
            signal=signal,
            entry_features=signal.features,
        )

        if config.trading.dry_run:
            return self._dry_run(position)

        return await self._live_trade(position)

    async def sync_open_trades(self):
        if config.trading.dry_run:
            return

        try:
            live_trades = await self._oanda.get_open_trades()
            live_ids = {t.get("id") for t in live_trades}

            for trade_id, pos in list(self._open_positions.items()):
                oanda_id = getattr(pos, "_oanda_trade_id", None)
                if oanda_id and oanda_id not in live_ids:
                    await self._handle_closed_trade(pos, oanda_id)

        except Exception as e:
            logger.warning(f"[Executor] Sync failed: {e}")

    async def check_dry_run_exits(self, current_prices: dict[str, dict]):
        for trade_id, pos in list(self._open_positions.items()):
            price_data = current_prices.get(pos.instrument)
            if not price_data:
                continue

            mid = price_data.get("mid", 0)
            if mid <= 0:
                continue

            exit_price = None
            exit_reason = ""

            if pos.direction == Direction.LONG:
                if mid <= pos.stop_loss:
                    exit_price = pos.stop_loss
                    exit_reason = "stop_loss"
                elif mid >= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = "take_profit"
            else:
                if mid >= pos.stop_loss:
                    exit_price = pos.stop_loss
                    exit_reason = "stop_loss"
                elif mid <= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = "take_profit"

            if exit_price is not None:
                self._close_position(pos, exit_price, exit_reason)

    def get_open_positions(self) -> dict[str, Position]:
        return self._open_positions.copy()

    def get_open_count(self, instrument: Optional[str] = None) -> int:
        if instrument:
            return sum(1 for p in self._open_positions.values() if p.instrument == instrument)
        return len(self._open_positions)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _dry_run(self, position: Position) -> Position:
        self._open_positions[position.trade_id] = position
        self._store.record_trade(position, dry_run=True)
        logger.info(
            f"[Executor] PAPER {position.direction.value.upper()} "
            f"{position.instrument} {abs(position.size_units):.0f} units "
            f"@ {position.entry_price:.5f} | "
            f"SL={position.stop_loss:.5f} TP={position.take_profit:.5f} | "
            f"risk=${position.risk_usdc:.2f}"
        )
        return position

    async def _live_trade(self, position: Position) -> Optional[Position]:
        try:
            data = await self._oanda.place_market_order(
                instrument=position.instrument,
                units=int(position.size_units),
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
            )

            fill = data.get("orderFillTransaction")
            if fill:
                actual_price = float(fill.get("price", position.entry_price))
                position.entry_price = actual_price
                oanda_trade_id = fill.get("tradeOpened", {}).get("tradeID", "")
                setattr(position, "_oanda_trade_id", oanda_trade_id)

                self._open_positions[position.trade_id] = position
                self._store.record_trade(position, dry_run=False)

                logger.info(
                    f"[Executor] LIVE {position.direction.value.upper()} "
                    f"{position.instrument} {abs(position.size_units):.0f} units "
                    f"@ {actual_price:.5f} | OANDA ID: {oanda_trade_id}"
                )
                return position

            cancel = data.get("orderCancelTransaction", {})
            logger.warning(f"[Executor] Order rejected: {cancel.get('reason', 'unknown')}")
            return None

        except Exception as e:
            logger.error(f"[Executor] Live trade failed: {e}")
            return None

    async def _handle_closed_trade(self, position: Position, oanda_trade_id: str):
        price_data = await self._oanda.get_current_price(position.instrument)
        exit_price = price_data.get("mid", position.entry_price)
        self._close_position(position, exit_price, "broker_closed")

    def _close_position(self, position: Position, exit_price: float, reason: str):
        pip = _pip_size(position.instrument)

        if position.direction == Direction.LONG:
            pnl = (exit_price - position.entry_price) * abs(position.size_units)
            pnl_pips = (exit_price - position.entry_price) / pip
        else:
            pnl = (position.entry_price - exit_price) * abs(position.size_units)
            pnl_pips = (position.entry_price - exit_price) / pip

        position.exit_price = exit_price
        position.pnl = round(pnl, 2)
        position.pnl_pips = round(pnl_pips, 1)
        position.exit_reason = reason
        position.status = TradeStatus.CLOSED
        position.closed_at = datetime.now(timezone.utc)

        self._store.close_trade(
            trade_id=position.trade_id,
            exit_price=exit_price,
            pnl=round(pnl, 2),
            pnl_pips=round(pnl_pips, 1),
            exit_reason=reason,
        )

        del self._open_positions[position.trade_id]

        icon = "+" if pnl > 0 else ""
        logger.info(
            f"[Executor] CLOSED {position.instrument} "
            f"{position.direction.value.upper()} | {reason} | "
            f"P&L: {icon}${pnl:.2f} ({icon}{pnl_pips:.1f} pips)"
        )
