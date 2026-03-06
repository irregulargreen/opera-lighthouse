"""
backtest.py — historical simulation engine.

Replays historical candles through the full strategy + ML pipeline.
Uses the same code paths as live trading for accuracy.

Usage:
  python backtest.py                          # Default: EUR_USD H1, last 2 years
  python backtest.py --instrument GBP_USD     # Different pair
  python backtest.py --days 365               # Last year
  python backtest.py --all                    # All configured instruments
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.config import config
from core.instrument_params import get_params
from core.models import Direction, PriceSeries, Regime, Signal
from data.feature_engine import FeatureEngine
from data.market_data import OANDAClient
from ml.regime_detector import RegimeDetector
from ml.signal_combiner import SignalCombiner
from strategies.breakout import Breakout
from strategies.mean_reversion import MeanReversion
from strategies.trend_following import TrendFollowing

console = Console()


class BacktestEngine:
    """
    Walk-forward backtester.

    Processes candles one at a time, simulating real-time conditions.
    No look-ahead bias: indicators and ML models only see data up to
    the current bar.
    """

    def __init__(self):
        self.features = FeatureEngine()
        self.strategies = [TrendFollowing(), MeanReversion(), Breakout()]

        self.trades: list[dict] = []
        self.equity_curve: list[float] = []
        self.open_trades: list[dict] = []

    async def run(
        self,
        instrument: str = "EUR_USD",
        timeframe: str = "H1",
        days: int = 730,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.02,
    ):
        console.print(Panel.fit(
            f"[bold cyan]Backtesting {instrument}[/bold cyan]\n"
            f"[dim]{timeframe} over {days} days | ${initial_capital:,.0f} capital | "
            f"{risk_per_trade:.0%} risk per trade[/dim]",
            border_style="cyan",
        ))

        oanda = OANDAClient()
        params = get_params(instrument)

        # Fetch primary timeframe data
        console.print("[dim]Fetching H1 historical data...[/dim]")
        series = await self._fetch_full_history(oanda, instrument, timeframe, days)
        console.print(f"[dim]Loaded {len(series.candles)} {timeframe} candles[/dim]")

        if len(series.candles) < 250:
            console.print("[red]Not enough data for backtest (need >=250 candles)[/red]")
            return

        # Fetch H4 data for multi-timeframe analysis
        console.print("[dim]Fetching H4 data for MTF analysis...[/dim]")
        h4_series = await self._fetch_full_history(oanda, instrument, "H4", days)
        console.print(f"[dim]Loaded {len(h4_series.candles)} H4 candles[/dim]")

        h4_features_by_time: dict[datetime, dict] = {}
        if len(h4_series.candles) >= 50:
            h4_series = self.features.compute_indicators(h4_series)
            h4_features_by_time = self._precompute_h4_features(h4_series)

        capital = initial_capital
        peak_capital = capital
        max_drawdown = 0.0
        warmup = 200

        console.print("[dim]Computing indicators...[/dim]")
        series = self.features.compute_indicators(series)

        from core.models import PriceSeries as PS

        for i in range(warmup, len(series.candles)):
            feat = self._features_at_index(series, i)
            if not feat:
                continue

            # Add HTF bias from pre-computed H4 features
            htf_bias, htf_adx, htf_ma = self._get_htf_at_time(
                series.candles[i].timestamp, h4_features_by_time
            )
            feat["htf_bias"] = htf_bias
            feat["htf_adx"] = htf_adx
            feat["htf_ma_alignment"] = htf_ma

            # Cross-pair features default to neutral in single-instrument backtest
            feat["usd_strength"] = 0.0
            feat["market_vol_sync"] = 1.0

            current_price = series.candles[i].close
            current_high = series.candles[i].high
            current_low = series.candles[i].low

            # Check open trade exits
            for trade in list(self.open_trades):
                exit_price = None
                exit_reason = ""

                if trade["direction"] == Direction.LONG:
                    if current_low <= trade["stop_loss"]:
                        exit_price = trade["stop_loss"]
                        exit_reason = "stop_loss"
                    elif current_high >= trade["take_profit"]:
                        exit_price = trade["take_profit"]
                        exit_reason = "take_profit"
                elif trade["direction"] == Direction.SHORT:
                    if current_high >= trade["stop_loss"]:
                        exit_price = trade["stop_loss"]
                        exit_reason = "stop_loss"
                    elif current_low <= trade["take_profit"]:
                        exit_price = trade["take_profit"]
                        exit_reason = "take_profit"

                if exit_price is not None:
                    if trade["direction"] == Direction.LONG:
                        pnl = (exit_price - trade["entry"]) * trade["units"]
                    else:
                        pnl = (trade["entry"] - exit_price) * trade["units"]

                    capital += pnl
                    trade["exit"] = exit_price
                    trade["pnl"] = pnl
                    trade["exit_reason"] = exit_reason
                    trade["exit_bar"] = i
                    self.trades.append(trade)
                    self.open_trades.remove(trade)

            self.equity_curve.append(capital)
            if capital > peak_capital:
                peak_capital = capital
            dd = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, dd)

            if len(self.open_trades) >= config.trading.max_open_positions:
                continue

            regime = self.features.classify_regime(feat)

            window = PS(
                instrument=instrument,
                timeframe=timeframe,
                candles=series.candles[:i + 1],
                sma_20=series.sma_20[:i + 1] if series.sma_20 else None,
                sma_50=series.sma_50[:i + 1] if series.sma_50 else None,
                sma_200=series.sma_200[:i + 1] if series.sma_200 else None,
                ema_12=series.ema_12[:i + 1] if series.ema_12 else None,
                ema_26=series.ema_26[:i + 1] if series.ema_26 else None,
                rsi_14=series.rsi_14[:i + 1] if series.rsi_14 else None,
                macd_line=series.macd_line[:i + 1] if series.macd_line else None,
                macd_signal=series.macd_signal[:i + 1] if series.macd_signal else None,
                macd_histogram=series.macd_histogram[:i + 1] if series.macd_histogram else None,
                bb_upper=series.bb_upper[:i + 1] if series.bb_upper else None,
                bb_middle=series.bb_middle[:i + 1] if series.bb_middle else None,
                bb_lower=series.bb_lower[:i + 1] if series.bb_lower else None,
                atr_14=series.atr_14[:i + 1] if series.atr_14 else None,
                adx_14=series.adx_14[:i + 1] if series.adx_14 else None,
                donchian_upper=series.donchian_upper[:i + 1] if series.donchian_upper else None,
                donchian_lower=series.donchian_lower[:i + 1] if series.donchian_lower else None,
                returns=series.returns[:i + 1] if series.returns else None,
                volatility_20=series.volatility_20[:i + 1] if series.volatility_20 else None,
            )

            for strategy in self.strategies:
                sig = strategy.evaluate(window, feat, regime)
                if not sig or sig.direction == Direction.FLAT:
                    continue

                if sig.strength < 0.45:
                    continue

                if any(t["instrument"] == instrument for t in self.open_trades):
                    continue

                risk_amount = capital * risk_per_trade
                risk_distance = abs(sig.entry_price - sig.stop_loss)
                if risk_distance <= 0:
                    continue
                units = risk_amount / risk_distance

                self.open_trades.append({
                    "instrument": instrument,
                    "direction": sig.direction,
                    "strategy": sig.source.value,
                    "entry": current_price,
                    "stop_loss": sig.stop_loss,
                    "take_profit": sig.take_profit,
                    "units": units,
                    "risk": risk_amount,
                    "regime": regime.value,
                    "entry_bar": i,
                    "signal_strength": sig.strength,
                    "htf_bias": htf_bias,
                })
                break

        # Close remaining open trades at last price
        last_price = series.candles[-1].close
        for trade in self.open_trades:
            if trade["direction"] == Direction.LONG:
                pnl = (last_price - trade["entry"]) * trade["units"]
            else:
                pnl = (trade["entry"] - last_price) * trade["units"]
            trade["exit"] = last_price
            trade["pnl"] = pnl
            trade["exit_reason"] = "end_of_data"
            self.trades.append(trade)

        self._print_results(instrument, initial_capital, capital, max_drawdown, days)

    # ── Multi-timeframe helpers ───────────────────────────────────────────────

    def _precompute_h4_features(self, h4_series: PriceSeries) -> dict[datetime, dict]:
        """Pre-compute features for each H4 bar."""
        result: dict[datetime, dict] = {}
        for i in range(50, len(h4_series.candles)):
            feat = self._features_at_index(h4_series, i)
            if feat:
                result[h4_series.candles[i].timestamp] = feat
        return result

    @staticmethod
    def _get_htf_at_time(
        h1_time: datetime,
        h4_features: dict[datetime, dict],
    ) -> tuple[float, float, float]:
        """Find the most recent closed H4 bar before this H1 timestamp."""
        if not h4_features:
            return 0.0, 20.0, 0.0

        # Find latest H4 timestamp <= h1_time
        best_time = None
        for t in h4_features:
            if t <= h1_time:
                if best_time is None or t > best_time:
                    best_time = t

        if best_time is None:
            return 0.0, 20.0, 0.0

        h4_feat = h4_features[best_time]
        ma_align = h4_feat.get("ma_alignment", 0.0)
        adx = h4_feat.get("adx", 20.0)
        ret_20 = h4_feat.get("ret_20", 0.0)

        if adx < 18:
            bias = 0.0
        else:
            bias = ma_align * 0.6
            adx_factor = min(1.0, (adx - 18) / 30)
            bias *= (0.5 + 0.5 * adx_factor)
            if ret_20 != 0:
                momentum_sign = 1.0 if ret_20 > 0 else -1.0
                bias += momentum_sign * min(0.2, abs(ret_20) * 5.0)
            bias = max(-1.0, min(1.0, bias))

        return round(bias, 4), adx, ma_align

    # ── Feature extraction ────────────────────────────────────────────────────

    def _features_at_index(self, series, idx: int) -> dict[str, float]:
        if idx < 50:
            return {}

        c = series.closes[:idx + 1]
        price = c[-1]
        n = len(c)

        def _val(arr, default=0.0):
            if arr and idx < len(arr):
                v = arr[idx]
                return float(v) if v is not None and not np.isnan(v) else default
            return default

        sma20 = _val(series.sma_20, price)
        sma50 = _val(series.sma_50, price)
        sma200 = _val(series.sma_200, price)
        rsi = _val(series.rsi_14, 50.0)
        macd_hist = _val(series.macd_histogram)
        atr = _val(series.atr_14, price * 0.01)
        adx = _val(series.adx_14, 20.0)
        bb_upper = _val(series.bb_upper, price)
        bb_lower = _val(series.bb_lower, price)
        bb_mid = _val(series.bb_middle, price)
        vol = _val(series.volatility_20, 0.01)
        dc_upper = _val(series.donchian_upper, price)
        dc_lower = _val(series.donchian_lower, price)

        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0
        bb_pct = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
        dc_width = (dc_upper - dc_lower) / price if price > 0 else 0
        dc_pct = (price - dc_lower) / (dc_upper - dc_lower) if (dc_upper - dc_lower) > 0 else 0.5

        ma_alignment = 0.0
        if price > sma20:
            ma_alignment += 0.33
        if sma20 > sma50:
            ma_alignment += 0.33
        if sma50 > sma200:
            ma_alignment += 0.34
        if price < sma20 < sma50 < sma200:
            ma_alignment = -1.0
        elif price < sma20:
            ma_alignment = -(abs(ma_alignment))

        ret_5 = (price - c[-6]) / c[-6] if n > 5 else 0.0
        ret_20 = (price - c[-21]) / c[-21] if n > 20 else 0.0

        if series.atr_14 and idx >= 50:
            atr_vals = [v for v in series.atr_14[idx - 49:idx + 1] if v is not None and not np.isnan(v)]
            atr_avg = float(np.mean(atr_vals)) if atr_vals else atr
            atr_ratio = atr / atr_avg if atr_avg > 0 else 1.0
        else:
            atr_ratio = 1.0

        dist_sma20 = (price - sma20) / atr if atr > 0 else 0
        dist_sma50 = (price - sma50) / atr if atr > 0 else 0

        vols = series.volumes[max(0, idx - 19):idx + 1]
        vol_recent = float(np.mean(vols[-5:])) if len(vols) >= 5 else 1.0
        vol_avg = float(np.mean(vols)) if len(vols) > 0 else 1.0
        volume_ratio = vol_recent / vol_avg if vol_avg > 0 else 1.0

        return {
            "rsi": rsi, "adx": adx,
            "macd_histogram": macd_hist / atr if atr > 0 else 0,
            "ma_alignment": ma_alignment,
            "bb_pct": bb_pct, "bb_width": bb_width,
            "dc_pct": dc_pct, "dc_width": dc_width,
            "atr_ratio": atr_ratio, "volatility": vol,
            "dist_sma20": dist_sma20, "dist_sma50": dist_sma50,
            "ret_5": ret_5, "ret_20": ret_20,
            "volume_ratio": volume_ratio,
        }

    @staticmethod
    async def _fetch_full_history(
        oanda: OANDAClient,
        instrument: str,
        timeframe: str,
        days: int,
    ) -> PriceSeries:
        from core.models import PriceSeries as PS
        import asyncio as _asyncio

        all_candles = []
        chunk_size = 4000
        current_from = datetime.now(timezone.utc) - timedelta(days=days)

        while True:
            try:
                series = await oanda.get_candles(
                    instrument, timeframe, count=chunk_size,
                    from_time=current_from,
                )
                if not series.candles:
                    break
                all_candles.extend(series.candles)
                current_from = series.candles[-1].timestamp + timedelta(seconds=1)
                if len(series.candles) < chunk_size:
                    break
                await _asyncio.sleep(0.3)
            except Exception as e:
                logger.warning(f"[Backtest] Fetch chunk failed: {e}")
                break

        seen = set()
        unique = []
        for c in all_candles:
            if c.timestamp not in seen:
                seen.add(c.timestamp)
                unique.append(c)
        unique.sort(key=lambda c: c.timestamp)

        return PS(instrument=instrument, timeframe=timeframe, candles=unique)

    def _print_results(
        self,
        instrument: str,
        initial_capital: float,
        final_capital: float,
        max_drawdown: float,
        days: int,
    ):
        if not self.trades:
            console.print("[red]No trades generated during backtest[/red]")
            return

        pnls = [t["pnl"] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl = sum(pnls)
        total_return = (final_capital - initial_capital) / initial_capital

        strategy_stats: dict[str, dict] = {}
        for t in self.trades:
            strat = t.get("strategy", "unknown")
            if strat not in strategy_stats:
                strategy_stats[strat] = {"pnls": [], "wins": 0, "losses": 0}
            strategy_stats[strat]["pnls"].append(t["pnl"])
            if t["pnl"] > 0:
                strategy_stats[strat]["wins"] += 1
            else:
                strategy_stats[strat]["losses"] += 1

        regime_stats: dict[str, dict] = {}
        for t in self.trades:
            reg = t.get("regime", "unknown")
            if reg not in regime_stats:
                regime_stats[reg] = {"pnls": [], "count": 0}
            regime_stats[reg]["pnls"].append(t["pnl"])
            regime_stats[reg]["count"] += 1

        if len(pnls) > 1:
            sharpe = np.mean(pnls) / np.std(pnls, ddof=1) * np.sqrt(252) if np.std(pnls) > 0 else 0
        else:
            sharpe = 0

        console.print()
        console.print(Panel(
            f"[bold]Total Return: [{'green' if total_return >= 0 else 'red'}]"
            f"{total_return:+.1%}[/] (${total_pnl:+,.2f})[/bold]\n"
            f"Final Capital: ${final_capital:,.2f}\n"
            f"Total Trades: {len(self.trades)}\n"
            f"Win Rate: {len(wins)/len(self.trades):.1%}\n"
            f"Avg Win: ${np.mean(wins):.2f} | Avg Loss: ${np.mean(losses):.2f}\n"
            f"Profit Factor: {sum(wins)/abs(sum(losses)):.2f}\n"
            f"Sharpe Ratio: {sharpe:.2f}\n"
            f"Max Drawdown: {max_drawdown:.1%}\n"
            f"Trades/Month: {len(self.trades) / (days/30):.1f}",
            title=f"Backtest Results — {instrument}", border_style="cyan",
        ))

        table = Table(title="Strategy Breakdown", border_style="blue")
        table.add_column("Strategy")
        table.add_column("Trades", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("Avg Trade", justify="right")

        for strat, stats in strategy_stats.items():
            n = len(stats["pnls"])
            wr = stats["wins"] / n if n > 0 else 0
            total = sum(stats["pnls"])
            avg = np.mean(stats["pnls"])
            pnl_style = "green" if total >= 0 else "red"
            table.add_row(
                strat, str(n), f"{wr:.1%}",
                f"[{pnl_style}]${total:+,.2f}[/{pnl_style}]",
                f"${avg:+.2f}",
            )
        console.print(table)

        table2 = Table(title="Regime Breakdown", border_style="blue")
        table2.add_column("Regime")
        table2.add_column("Trades", justify="right")
        table2.add_column("Win Rate", justify="right")
        table2.add_column("P&L", justify="right")

        for reg, stats in regime_stats.items():
            n = stats["count"]
            wins_r = sum(1 for p in stats["pnls"] if p > 0)
            wr = wins_r / n if n > 0 else 0
            total = sum(stats["pnls"])
            pnl_style = "green" if total >= 0 else "red"
            table2.add_row(
                reg, str(n), f"{wr:.1%}",
                f"[{pnl_style}]${total:+,.2f}[/{pnl_style}]",
            )
        console.print(table2)

        exit_reasons: dict[str, int] = {}
        for t in self.trades:
            r = t.get("exit_reason", "unknown")
            exit_reasons[r] = exit_reasons.get(r, 0) + 1
        console.print(f"\n[dim]Exit reasons: {exit_reasons}[/dim]")


async def main():
    parser = argparse.ArgumentParser(description="Backtest the adaptive trading system")
    parser.add_argument("--instrument", "-i", default="EUR_USD", help="Instrument to backtest")
    parser.add_argument("--timeframe", "-t", default="H1", help="Candle timeframe")
    parser.add_argument("--days", "-d", type=int, default=730, help="Days of history")
    parser.add_argument("--capital", "-c", type=float, default=10000.0, help="Starting capital")
    parser.add_argument("--risk", "-r", type=float, default=0.02, help="Risk per trade (fraction)")
    parser.add_argument("--all", action="store_true", help="Backtest all configured instruments")
    args = parser.parse_args()

    if args.all:
        for inst in config.trading.instruments:
            engine = BacktestEngine()
            await engine.run(inst, args.timeframe, args.days, args.capital, args.risk)
            console.print()
    else:
        engine = BacktestEngine()
        await engine.run(args.instrument, args.timeframe, args.days, args.capital, args.risk)


if __name__ == "__main__":
    asyncio.run(main())
