"""
main.py — the trading bot loop.

Pipeline per cycle:
  1. Fetch candles for all instruments × primary + H4 timeframes
  2. Compute indicators → extract features → classify regime
  3. Compute cross-pair features (USD strength, volatility sync)
  4. Add higher-timeframe bias to features
  5. Run all strategies → collect signals
  6. ML signal combiner ranks and filters signals
  7. Risk manager approves/blocks
  8. Position sizer determines units
  9. Executor places trades
 10. Check for closed positions → feed outcomes back to ML

The system starts conservative (heuristic-only) and gets sharper
as resolved trades accumulate and the ML models activate.
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.add(
    LOG_DIR / "bot_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    compression="gz",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    level="DEBUG",
    enqueue=True,
)

from core.config import config
from core.models import Direction, PriceSeries, Regime, Signal
from data.feature_engine import FeatureEngine
from data.market_data import OANDAClient
from data.store import TradeStore
from execution.executor import Executor
from execution.risk_manager import RiskManager
from ml.position_sizer import PositionSizer
from ml.regime_detector import RegimeDetector
from ml.signal_combiner import SignalCombiner
from strategies.breakout import Breakout
from strategies.mean_reversion import MeanReversion
from strategies.trend_following import TrendFollowing

console = Console()


class TradingBot:

    def __init__(self, instrument_filter: str | None = None):
        self.oanda = OANDAClient()
        self.features = FeatureEngine()
        self.risk = RiskManager()

        self.strategies = [
            TrendFollowing(),
            MeanReversion(),
            Breakout(),
        ]

        self._running = False
        self._cycle_count = 0
        self._last_regime_label_time: dict[str, datetime] = {}

        # Per-instrument mode
        self._instrument_filter = instrument_filter
        if instrument_filter:
            self._instruments = [instrument_filter]
            suffix = f"_{instrument_filter}"
        else:
            self._instruments = config.trading.instruments
            suffix = ""

        # ML + persistence paths scoped by mode
        self.store = TradeStore(suffix=suffix)
        self.executor = Executor(self.store, self.oanda)
        self.regime_detector = RegimeDetector(suffix=suffix)
        self.signal_combiner = SignalCombiner(suffix=suffix)
        self.sizer = PositionSizer()

        # Cross-pair data cache (populated each cycle)
        self._pair_returns: dict[str, float] = {}
        self._pair_atr_ratios: dict[str, float] = {}

    async def start(self):
        self._print_startup()

        if not config.oanda.is_configured:
            logger.error("OANDA API not configured. Set OANDA_API_KEY and OANDA_ACCOUNT_ID in .env")
            sys.exit(1)

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._shutdown)

        self._running = True
        await self._loop()

    # ── Main Loop ────────────────────────────────────────────────────────────

    async def _loop(self):
        while self._running:
            cycle_start = datetime.now(timezone.utc)
            self._cycle_count += 1

            try:
                logger.info(f"{'=' * 60}")
                logger.info(f"[Bot] Cycle {self._cycle_count} @ {cycle_start.strftime('%H:%M:%S UTC')}")

                if config.trading.dry_run:
                    prices = {}
                    for inst in self._instruments:
                        try:
                            prices[inst] = await self.oanda.get_current_price(inst)
                        except Exception:
                            pass
                    await self.executor.check_dry_run_exits(prices)
                else:
                    await self.executor.sync_open_trades()

                self._process_closed_trades()

                # Phase 1: fetch data and compute features for ALL instruments
                instrument_data: dict[str, dict] = {}
                for instrument in self._instruments:
                    data = await self._fetch_instrument_data(instrument)
                    if data:
                        instrument_data[instrument] = data

                # Phase 2: compute cross-pair features
                cross_features = self._compute_cross_pair_features(instrument_data)

                # Phase 3: evaluate each instrument with full context
                all_signals: list[Signal] = []
                for instrument, data in instrument_data.items():
                    data["features"].update(cross_features)
                    signals = self._evaluate_instrument(data)
                    all_signals.extend(signals)

                if not all_signals:
                    logger.info("[Bot] No signals this cycle")
                else:
                    await self._process_signals(all_signals)

            except Exception as e:
                logger.error(f"[Bot] Cycle error: {e}", exc_info=True)

            elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
            sleep_time = max(10, 300 - elapsed)
            logger.info(f"[Bot] Cycle done in {elapsed:.1f}s — sleeping {sleep_time:.0f}s")
            await asyncio.sleep(sleep_time)

    # ── Data fetching ─────────────────────────────────────────────────────────

    async def _fetch_instrument_data(self, instrument: str) -> dict | None:
        """Fetch H1 + H4 candles, compute indicators and features."""
        timeframes = config.strategy.trend_timeframes
        primary_tf = timeframes[0] if timeframes else "H1"

        try:
            series = await self.oanda.get_candles(
                instrument, primary_tf, config.strategy.lookback_bars
            )
        except Exception as e:
            logger.warning(f"[Bot] Failed to fetch {instrument} {primary_tf}: {e}")
            return None

        if len(series.candles) < 50:
            return None

        series = self.features.compute_indicators(series)
        feat = self.features.extract_features(series)
        if not feat:
            return None

        # Fetch H4 for higher-timeframe bias
        htf_bias = 0.0
        htf_adx = 20.0
        htf_ma_align = 0.0
        try:
            h4_series = await self.oanda.get_candles(instrument, "H4", 200)
            if len(h4_series.candles) >= 50:
                h4_series = self.features.compute_indicators(h4_series)
                h4_feat = self.features.extract_features(h4_series)
                if h4_feat:
                    htf_ma_align = h4_feat.get("ma_alignment", 0.0)
                    htf_adx = h4_feat.get("adx", 20.0)
                    htf_bias = self._calculate_htf_bias(h4_feat)
        except Exception as e:
            logger.debug(f"[Bot] H4 fetch failed for {instrument}: {e}")

        feat["htf_bias"] = htf_bias
        feat["htf_adx"] = htf_adx
        feat["htf_ma_alignment"] = htf_ma_align

        # Cache for cross-pair computation
        self._pair_returns[instrument] = feat.get("ret_5", 0.0)
        self._pair_atr_ratios[instrument] = feat.get("atr_ratio", 1.0)

        regime, regime_confidence = self.regime_detector.predict(feat)
        logger.debug(
            f"[Bot] {instrument} regime: {regime.value} "
            f"(conf={regime_confidence:.0%}) HTF_bias={htf_bias:+.2f}"
        )

        # Feed regime labeler
        self._maybe_label_regime(instrument, feat, series)

        return {
            "instrument": instrument,
            "series": series,
            "features": feat,
            "regime": regime,
        }

    @staticmethod
    def _calculate_htf_bias(h4_feat: dict[str, float]) -> float:
        """Convert H4 features into a directional bias score (-1 to +1)."""
        ma_align = h4_feat.get("ma_alignment", 0.0)
        adx = h4_feat.get("adx", 20.0)
        ret_20 = h4_feat.get("ret_20", 0.0)

        # ADX must show some directionality for bias to matter
        if adx < 18:
            return 0.0

        bias = 0.0

        # MA alignment is the primary signal
        bias += ma_align * 0.6

        # ADX scales the conviction
        adx_factor = min(1.0, (adx - 18) / 30)
        bias *= (0.5 + 0.5 * adx_factor)

        # Recent H4 momentum
        if ret_20 != 0:
            momentum_sign = 1.0 if ret_20 > 0 else -1.0
            bias += momentum_sign * min(0.2, abs(ret_20) * 5.0)

        return round(max(-1.0, min(1.0, bias)), 4)

    # ── Cross-pair features ───────────────────────────────────────────────────

    def _compute_cross_pair_features(self, instrument_data: dict[str, dict]) -> dict[str, float]:
        """Compute features that require data from multiple instruments."""
        if len(instrument_data) < 2:
            return {"usd_strength": 0.0, "market_vol_sync": 1.0}

        # USD strength: average directional move of USD across pairs
        usd_returns = []
        for inst, data in instrument_data.items():
            ret = data["features"].get("ret_5", 0.0)
            if inst.startswith("USD_"):
                usd_returns.append(ret)
            elif inst.endswith("_USD") and inst != "XAU_USD":
                usd_returns.append(-ret)

        usd_strength = float(np.mean(usd_returns)) if usd_returns else 0.0

        # Market volatility synchronization
        atr_ratios = [
            data["features"].get("atr_ratio", 1.0)
            for data in instrument_data.values()
        ]
        market_vol_sync = float(np.mean(atr_ratios)) if atr_ratios else 1.0

        return {
            "usd_strength": round(usd_strength, 6),
            "market_vol_sync": round(market_vol_sync, 4),
        }

    # ── Per-instrument evaluation ────────────────────────────────────────────

    def _evaluate_instrument(self, data: dict) -> list[Signal]:
        """Run all strategies on pre-computed instrument data."""
        series = data["series"]
        feat = data["features"]
        regime = data["regime"]

        signals: list[Signal] = []
        for strategy in self.strategies:
            try:
                sig = strategy.evaluate(series, feat, regime)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.warning(f"[Bot] Strategy {strategy.name} error: {e}")

        return signals

    # ── Signal processing and execution ──────────────────────────────────────

    async def _process_signals(self, signals: list[Signal]):
        regime = signals[0].regime if signals else Regime.UNKNOWN

        ranked = self.signal_combiner.rank_signals(signals, regime)

        try:
            account = await self.oanda.get_account_summary()
            capital = account.get("nav", config.trading.total_capital)
        except Exception:
            capital = config.trading.total_capital

        open_positions = self.executor.get_open_positions()

        executed = 0
        for signal, score in ranked:
            should_trade, reason = self.signal_combiner.should_trade(signal, score)
            if not should_trade:
                logger.debug(f"[Bot] Signal blocked: {reason}")
                continue

            allowed, risk_reason = self.risk.check(signal, open_positions, capital)
            if not allowed:
                logger.debug(f"[Bot] Risk blocked: {risk_reason}")
                continue

            perf = self.store.get_performance(strategy=signal.source.value, last_n=50)
            win_prob = perf.win_rate if perf.total_trades >= 20 else 0.5
            rr_ratio = (perf.avg_win / abs(perf.avg_loss)) if perf.avg_loss != 0 else 1.5

            size_info = self.sizer.calculate(
                signal=signal,
                capital=capital,
                win_prob=win_prob,
                avg_win_loss_ratio=rr_ratio,
            )

            if size_info["units"] == 0:
                continue

            position = await self.executor.execute(
                signal=signal,
                units=size_info["units"],
                risk_usdc=size_info["risk_usdc"],
            )

            if position:
                executed += 1
                open_positions[position.trade_id] = position
                self._print_trade(signal, score, size_info)

        if executed > 0:
            logger.info(f"[Bot] Executed {executed} trade(s) this cycle")

    # ── ML feedback loop ─────────────────────────────────────────────────────

    def _process_closed_trades(self):
        closed = self.store.get_closed_trades(limit=10)
        for trade in closed:
            if trade.get("_ml_processed"):
                continue

            pnl = trade.get("pnl")
            if pnl is None:
                continue

            won = pnl > 0
            strategy = trade.get("strategy", "unknown")
            regime_str = trade.get("regime", "unknown")

            try:
                regime = Regime(regime_str)
            except ValueError:
                regime = Regime.UNKNOWN

            signal = Signal(
                instrument=trade.get("instrument", ""),
                source=next(
                    (s for s in [
                        "trend_following", "mean_reversion", "breakout"
                    ] if s == strategy),
                    "trend_following",
                ),
                direction=Direction(trade.get("direction", "long")),
                strength=trade.get("signal_strength", 0.5),
                confidence=trade.get("signal_confidence", 0.5),
                regime=regime,
                features=self._parse_features(trade.get("entry_features", "{}")),
            )

            self.signal_combiner.record_outcome(signal, won, pnl, regime)
            self.risk.record_daily_pnl(pnl)

    def _maybe_label_regime(self, instrument: str, features: dict, series):
        now = datetime.now(timezone.utc)
        last = self._last_regime_label_time.get(instrument)
        if last and (now - last).total_seconds() < 7200:
            return

        if len(series.candles) < 40:
            return

        past_features = self.features.extract_features(
            type(series)(
                instrument=series.instrument,
                timeframe=series.timeframe,
                candles=series.candles[:-20],
                sma_20=series.sma_20[:-20] if series.sma_20 else None,
                sma_50=series.sma_50[:-20] if series.sma_50 else None,
                sma_200=series.sma_200[:-20] if series.sma_200 else None,
                ema_12=series.ema_12[:-20] if series.ema_12 else None,
                ema_26=series.ema_26[:-20] if series.ema_26 else None,
                rsi_14=series.rsi_14[:-20] if series.rsi_14 else None,
                macd_line=series.macd_line[:-20] if series.macd_line else None,
                macd_signal=series.macd_signal[:-20] if series.macd_signal else None,
                macd_histogram=series.macd_histogram[:-20] if series.macd_histogram else None,
                bb_upper=series.bb_upper[:-20] if series.bb_upper else None,
                bb_middle=series.bb_middle[:-20] if series.bb_middle else None,
                bb_lower=series.bb_lower[:-20] if series.bb_lower else None,
                atr_14=series.atr_14[:-20] if series.atr_14 else None,
                adx_14=series.adx_14[:-20] if series.adx_14 else None,
                donchian_upper=series.donchian_upper[:-20] if series.donchian_upper else None,
                donchian_lower=series.donchian_lower[:-20] if series.donchian_lower else None,
                returns=series.returns[:-20] if series.returns else None,
                volatility_20=series.volatility_20[:-20] if series.volatility_20 else None,
            )
        )

        if not past_features:
            return

        subsequent_closes = [c.close for c in series.candles[-20:]]
        atr = features.get("atr_ratio", 1.0) * series.closes[-1] * 0.01
        if series.atr_14:
            valid_atr = [v for v in series.atr_14[-25:-20] if v is not None]
            if valid_atr:
                atr = valid_atr[-1]

        self.regime_detector.label_from_hindsight(past_features, subsequent_closes, atr)
        self._last_regime_label_time[instrument] = now

    @staticmethod
    def _parse_features(features_str: str) -> dict[str, float]:
        try:
            import json
            return json.loads(features_str)
        except Exception:
            return {}

    # ── Display ──────────────────────────────────────────────────────────────

    def _print_startup(self):
        mode = "[yellow]PAPER TRADING[/yellow]" if config.trading.dry_run else "[red bold]LIVE[/red bold]"
        inst_mode = f"[magenta]single: {self._instrument_filter}[/magenta]" if self._instrument_filter else "multi"
        console.print(Panel.fit(
            "[bold cyan]Adaptive Forex Trader[/bold cyan]\n"
            "[dim]Multi-strategy system with ML regime detection[/dim]",
            border_style="cyan",
        ))
        console.print(f"Mode: {mode} | Instrument mode: {inst_mode}")
        console.print(f"Instruments: [cyan]{', '.join(self._instruments)}[/cyan]")
        console.print(f"Capital: [cyan]${config.trading.total_capital:,.2f}[/cyan]")
        console.print(f"Risk per trade: [cyan]{config.trading.max_risk_per_trade:.1%}[/cyan]")
        console.print(f"Kelly fraction: [cyan]{config.trading.kelly_fraction:.0%}[/cyan]")
        console.print(f"Strategies: [cyan]{', '.join(s.name for s in self.strategies)}[/cyan]")

        regime_status = self.regime_detector.get_status()
        if regime_status["is_fitted"]:
            console.print(f"Regime detector: [green]fitted[/green] (accuracy={regime_status['accuracy']:.1%})")
        else:
            console.print(
                f"Regime detector: [yellow]heuristic[/yellow] "
                f"(need {regime_status['needs']} more samples)"
            )

        combiner_insights = self.signal_combiner.get_insights()
        if combiner_insights["is_fitted"]:
            console.print(f"Signal combiner: [green]fitted[/green] (AUC={combiner_insights['model_auc']:.3f})")
        else:
            console.print(
                f"Signal combiner: [yellow]heuristic[/yellow] "
                f"({combiner_insights['training_samples']} samples)"
            )

        perf = self.store.get_performance()
        if perf.total_trades > 0:
            console.print(Panel(
                f"Trades: {perf.total_trades} | "
                f"Win rate: {perf.win_rate:.1%} | "
                f"P&L: {'$' if perf.total_pnl >= 0 else '-$'}{abs(perf.total_pnl):.2f} | "
                f"Sharpe: {perf.sharpe_ratio:.2f} | "
                f"Max DD: ${perf.max_drawdown:.2f}",
                title="Historical Performance", border_style="blue",
            ))
        console.print()

    def _print_trade(self, signal: Signal, score: float, size_info: dict):
        table = Table(title="Trade Executed", border_style="green" if signal.direction == Direction.LONG else "red")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Instrument", signal.instrument)
        table.add_row("Direction", signal.direction.value.upper())
        table.add_row("Strategy", signal.source.value)
        table.add_row("Signal Score", f"{score:.3f}")
        table.add_row("Entry", f"{signal.entry_price:.5f}")
        table.add_row("Stop Loss", f"{signal.stop_loss:.5f}")
        table.add_row("Take Profit", f"{signal.take_profit:.5f}")
        table.add_row("Units", f"{abs(size_info['units']):,.0f}")
        table.add_row("Risk", f"${size_info['risk_usdc']:.2f}")
        table.add_row("Regime", signal.regime.value)
        table.add_row("Reasoning", signal.reasoning)
        console.print(table)

    def _shutdown(self, *_):
        logger.info("[Bot] Shutting down...")
        self._running = False

        perf = self.store.get_performance()
        if perf.total_trades > 0:
            console.print(Panel(
                f"Total trades: {perf.total_trades}\n"
                f"Win rate: {perf.win_rate:.1%}\n"
                f"Total P&L: ${perf.total_pnl:.2f}\n"
                f"Profit factor: {perf.profit_factor:.2f}\n"
                f"Sharpe: {perf.sharpe_ratio:.2f}",
                title="Session Summary", border_style="blue",
            ))

        insights = self.signal_combiner.get_insights()
        if insights.get("strategy_regime_win_rates"):
            console.print("[bold]Strategy-Regime Win Rates:[/bold]")
            for key, stats in insights["strategy_regime_win_rates"].items():
                console.print(f"  {key}: {stats['win_rate']:.1%} ({stats['trades']} trades)")

        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Forex Trader")
    parser.add_argument(
        "--instrument", "-i",
        help="Run in single-instrument mode (e.g. EUR_USD)",
    )
    args = parser.parse_args()

    bot = TradingBot(instrument_filter=args.instrument)
    asyncio.run(bot.start())
