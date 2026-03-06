"""
main.py — the main bot loop.

Pipeline:
  Every SCAN_INTERVAL_SECONDS:
    1. Fetch new markets (high-priority, first-mover window)
    2. Enrich with news
    3. LLM estimate
    4. Calculate edge
    5. Execute if edge > threshold

  Every FULL_SWEEP_INTERVAL_SECONDS:
    Same pipeline but on ALL active markets in the "sweet spot" liquidity band

  On startup: print current stats from calibration DB
"""
from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from core.config import config
from core.models import EnrichedMarket, TradeSignal
from data.scanner import MarketScanner
from data.news_enricher import NewsEnricher
from strategies.tiered_forecaster import TieredForecaster
from strategies.edge_calculator import EdgeCalculator
from strategies.executor import Executor
from utils.calibration import CalibrationTracker
from utils.ml_market_classifier import MarketClassifier
from utils.meta_learner import MetaLearner


console = Console()


class PolymarketBot:
    """
    Full AI-powered Polymarket scanning and trading bot.
    Combines: market discovery → news enrichment → LLM forecasting →
              edge calculation → execution → calibration tracking.
    """

    def __init__(self):
        self.scanner = MarketScanner()
        self.enricher = NewsEnricher()
        self.forecaster = TieredForecaster()
        self.edge_calc = EdgeCalculator()
        self.executor = Executor()
        self.calibration = CalibrationTracker()
        self.market_classifier = MarketClassifier(min_edge=config.bot.min_edge)
        self.meta_learner = MetaLearner()
        self._running = False
        self._last_full_sweep = datetime.min
        self._processed_market_ids: set[str] = set()  # Avoid re-processing

    # ─── Startup ──────────────────────────────────────────────────────────────

    async def start(self):
        """Initialise components and start the main loop."""
        console.print(Panel.fit(
            "[bold green]Polymarket AI Scanner[/bold green]\n"
            "[dim]LLM-powered edge detection across prediction markets[/dim]",
            border_style="green",
        ))

        mode = "[yellow]DRY RUN[/yellow]" if config.bot.dry_run else "[red bold]LIVE TRADING[/red bold]"
        console.print(f"Mode: {mode}")
        console.print(f"Min edge: [cyan]{config.bot.min_edge:.1%}[/cyan]")
        console.print(f"Max trade: [cyan]${config.bot.max_trade_size_usdc:.2f}[/cyan]")
        console.print(f"Capital: [cyan]${config.bot.total_capital_usdc:.2f}[/cyan]")
        console.print(f"Kelly fraction: [cyan]{config.bot.kelly_fraction:.0%}[/cyan]")

        # Show enabled LLMs
        llms = []
        if config.llm.openai_enabled:
            llms.append("OpenAI GPT-4o")
        if config.llm.anthropic_enabled:
            llms.append("Claude Sonnet")
        if config.llm.ollama_enabled:
            llms.append(f"Ollama ({config.llm.ollama_model})")
        console.print(f"LLMs: [cyan]{', '.join(llms) or 'None configured!'}[/cyan]\n")

        if not llms:
            logger.error("No LLM providers configured! Set API keys in .env")
            sys.exit(1)

        # Show ML status
        ml_report = self.forecaster.get_ml_calibration_report()
        ml_status = f"[cyan]{ml_report['status']}[/cyan] ({ml_report['total_samples']} samples)"
        if ml_report['total_samples'] < 30:
            ml_status += f" [dim]— need {30 - ml_report['total_samples']} more resolved trades[/dim]"
        console.print(f"ML calibration: {ml_status}")

        clf_status = "fitted" if self.market_classifier._is_fitted else f"heuristic ({len(self.market_classifier._training_data)} samples)"
        console.print(f"Market classifier: [cyan]{clf_status}[/cyan]")

        meta_insights = self.meta_learner.get_insights()
        meta_status = (
            f"[cyan]{meta_insights['status']}[/cyan] "
            f"({meta_insights['total_resolved_trades']} resolved trades"
            + (f", AUC={meta_insights['model_auc']:.3f}" if meta_insights.get('model_auc') else "")
            + ")"
        )
        console.print(f"Meta-learner:     {meta_status}\n")

        # Initialise executor (connects to Polymarket CLOB)
        self.executor.initialise()

        # Register shutdown handler
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_shutdown)

        self._running = True
        await self._main_loop()

    # ─── Main loop ────────────────────────────────────────────────────────────

    async def _main_loop(self):
        async with self.scanner as scanner, self.enricher as enricher:
            self.scanner = scanner
            self.enricher = enricher

            while self._running:
                now = datetime.utcnow()
                cycle_start = datetime.utcnow()

                try:
                    # ── New market sweep (every SCAN_INTERVAL_SECONDS) ────────
                    logger.info("=" * 60)
                    logger.info(f"[Loop] New market scan @ {now.strftime('%H:%M:%S')}")

                    new_markets = await self.scanner.get_new_markets()
                    fresh = [m for m in new_markets if m.question_id not in self._processed_market_ids]

                    if fresh:
                        logger.info(f"[Loop] {len(fresh)} fresh markets to analyse")
                        await self._process_markets(fresh, priority="NEW")
                    else:
                        logger.info("[Loop] No new markets since last scan")

                    # ── Full sweep (every FULL_SWEEP_INTERVAL_SECONDS) ────────
                    seconds_since_sweep = (now - self._last_full_sweep).total_seconds()
                    if seconds_since_sweep >= config.bot.full_sweep_interval_seconds:
                        logger.info(f"[Loop] Full market sweep")
                        active = await self.scanner.get_active_markets(
                            min_volume_24h=500.0,
                            min_liquidity=config.bot.min_liquidity_usdc,
                            limit=300,
                        )
                        # Exclude markets already processed in recent history
                        to_process = [m for m in active if m.question_id not in self._processed_market_ids]
                        logger.info(f"[Loop] Full sweep: {len(to_process)} markets to analyse")

                        if to_process:
                            # Enrich order books concurrently before LLM evaluation
                            to_process = await self.scanner.enrich_with_order_books(to_process)
                            await self._process_markets(to_process, priority="SWEEP")

                        self._last_full_sweep = now

                except Exception as e:
                    logger.error(f"[Loop] Cycle error: {e}", exc_info=True)

                # ── Sleep until next cycle ────────────────────────────────────
                elapsed = (datetime.utcnow() - cycle_start).total_seconds()
                sleep_time = max(1, config.bot.scan_interval_seconds - elapsed)
                logger.info(f"[Loop] Cycle done in {elapsed:.1f}s — sleeping {sleep_time:.0f}s")
                await asyncio.sleep(sleep_time)

    # ─── Per-market pipeline ──────────────────────────────────────────────────

    async def _process_markets(self, markets: list, priority: str):
        """
        Run the full pipeline on a list of markets:
        news enrichment → LLM forecast → edge calculation → execution
        """
        signals: list[TradeSignal] = []

        # Step 1: Enrich with news (batch, concurrent)
        logger.info(f"[Pipeline] Enriching {len(markets)} markets with news...")
        enriched_list: list[EnrichedMarket] = await self.enricher.enrich_batch(markets)

        # Step 1b: ML pre-filter — score each market before any LLM call
        # Classifier uses gradient boosting on market features (<1ms, free)
        # Falls back to heuristic rules until model is trained
        pre_filtered = []
        ml_skipped = 0
        for enriched in enriched_list:
            score = self.market_classifier.predict_mispricing_score(enriched.market)
            if score >= 0.25:  # 25% chance of mispricing → worth evaluating
                pre_filtered.append(enriched)
            else:
                ml_skipped += 1

        if ml_skipped > 0:
            logger.info(
                f"[Pipeline] ML pre-filter: {ml_skipped} markets skipped (low mispricing score), "
                f"{len(pre_filtered)} proceeding to LLM"
            )

        # Step 2: LLM forecasting (respect API rate limits with semaphore)
        sem = asyncio.Semaphore(3)  # 3 concurrent LLM calls
        processed = 0

        async def forecast_and_evaluate(enriched: EnrichedMarket):
            nonlocal processed
            async with sem:
                try:
                    # Get LLM ensemble estimate
                    ensemble = await self.forecaster.estimate(enriched)
                    if ensemble is None:
                        return

                    # Calculate edge and generate signal
                    signal = self.edge_calc.evaluate(enriched, ensemble)
                    if signal:
                        signal.enriched_context = enriched  # Attach for meta-learner
                        signals.append(signal)

                except Exception as e:
                    logger.warning(f"[Pipeline] Error on '{enriched.market.question[:50]}': {e}")
                finally:
                    processed += 1
                    if processed % 10 == 0:
                        logger.info(f"[Pipeline] Processed {processed}/{len(enriched_list)} markets...")

        await asyncio.gather(*[forecast_and_evaluate(e) for e in pre_filtered])

        # Mark all as processed (even if no signal found — avoid re-evaluating)
        for m in markets:
            self._processed_market_ids.add(m.question_id)

        # Limit set size to prevent memory growth
        if len(self._processed_market_ids) > 5000:
            # Keep the most recent 3000
            self._processed_market_ids = set(list(self._processed_market_ids)[-3000:])

        # Step 3: Rank signals, meta-learner gate, adaptive Kelly, execute
        if not signals:
            logger.info(f"[Pipeline] {priority}: No edges found in this batch")
            return

        ranked = EdgeCalculator.rank_signals(signals)

        # Meta-learner final gate: filters out trades where historical patterns
        # suggest the LLM edge won't hold up, and adjusts Kelly sizing
        approved: list[TradeSignal] = []
        for signal in ranked:
            enriched = signal.enriched_context
            if enriched is None:
                approved.append(signal)
                continue
            should_trade, reason = self.meta_learner.should_trade(signal, enriched)
            kelly_mult = self.meta_learner.adjusted_kelly_multiplier(signal, enriched)
            if should_trade:
                signal.capped_size_usdc = round(
                    min(signal.capped_size_usdc * kelly_mult, config.bot.max_trade_size_usdc), 2
                )
                signal.expected_value = round(signal.edge_after_fees * signal.capped_size_usdc, 4)
                approved.append(signal)
                logger.debug(f"[Meta] ✅ kelly_mult={kelly_mult:.2f} | {reason}")
            else:
                logger.debug(f"[Meta] ❌ blocked | {reason}")

        if not approved:
            logger.info(f"[Pipeline] {priority}: All {len(ranked)} signals blocked by meta-learner")
            return

        logger.info(
            f"[Pipeline] {priority}: {len(ranked)} signals → "
            f"{len(approved)} approved by meta-learner"
        )
        self._print_signals(approved[:5])

        # Execute top approved signals
        for signal in approved:
            if len(self.executor.get_open_positions()) >= config.bot.max_open_positions:
                logger.info("[Pipeline] Max positions reached — stopping execution")
                break
            result = self.executor.execute(signal)
            if result.executed:
                self.calibration.record_trade(result)
                # Register with market classifier for training
                self.market_classifier.add_training_sample(
                    market=signal.market,
                    was_mispriced=True,  # We thought so — resolved outcome added later
                )

    def resolve_trade(self, order_id: str, won: bool, pnl: float):
        """
        Call this after a market resolves. Feeds outcome back to all ML layers.
        In production you'd hook this to a Polymarket webhook or poll for resolved markets.
        """
        from core.models import MarketOutcome
        outcome = MarketOutcome.YES if won else MarketOutcome.NO
        self.calibration.resolve_trade(order_id, outcome, pnl)

        # Find the signal in open positions to feed meta-learner
        position = self.executor.get_open_positions().get(order_id)
        if position and position.signal.enriched_context:
            self.meta_learner.record_outcome(
                signal=position.signal,
                enriched=position.signal.enriched_context,
                won=won,
                pnl=pnl,
            )
            # Feed market classifier too
            self.market_classifier.add_training_sample(
                market=position.signal.market,
                was_mispriced=won,  # If we won, our mispricing call was correct
                final_price=1.0 if won else 0.0,
            )
            # Periodically refit classifier
            if len(self.market_classifier._training_data) % 50 == 0:
                self.market_classifier.fit()

        logger.info(f"[Bot] Resolved {order_id}: {'WIN' if won else 'LOSS'} ${pnl:+.2f}")

    # ─── Display ──────────────────────────────────────────────────────────────

    def _print_signals(self, signals: list[TradeSignal]):
        if not signals:
            return

        table = Table(title=f"Top {len(signals)} Signals", border_style="cyan")
        table.add_column("Market", max_width=45, overflow="fold")
        table.add_column("Side", style="bold")
        table.add_column("Market P", justify="right")
        table.add_column("Our P", justify="right")
        table.add_column("Edge", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("EV", justify="right")
        table.add_column("Conf", justify="right")
        table.add_column("New?", justify="center")

        for s in signals:
            side_style = "green" if s.side.value == "YES" else "red"
            table.add_row(
                s.market.question[:45],
                f"[{side_style}]{s.side.value}[/{side_style}]",
                f"{s.market_price:.1%}",
                f"{s.estimated_probability:.1%}",
                f"[green]{s.edge_after_fees:.1%}[/green]",
                f"${s.capped_size_usdc:.2f}",
                f"${s.expected_value:.2f}",
                f"{s.ensemble.confidence:.0%}",
                "✓" if s.market.is_new else "",
            )

        console.print(table)

        costs = self.forecaster.get_cost_summary()
        if costs["total_usd"] > 0:
            console.print(Panel(
                f"Session LLM spend: [cyan]${costs['total_usd']:.4f}[/cyan]\n" +
                "\n".join(f"  {k}: ${v:.4f} ({costs['calls_by_provider'].get(k,0)} calls)"
                          for k, v in costs["by_provider"].items() if v > 0) +
                f"\n{costs['tip']}",
                title="Cost Tracking", border_style="yellow"
            ))

    def _print_stats(self):
        """Print performance stats from calibration DB."""
        stats = self.calibration.get_summary()
        if stats["total_trades"] == 0:
            console.print("[dim]No previous trade history found.[/dim]\n")
            return

        panel_text = (
            f"Total trades: [cyan]{stats['total_trades']}[/cyan] "
            f"({stats['dry_run_trades']} dry run)\n"
            f"Resolved: [cyan]{stats['resolved_trades']}[/cyan]\n"
        )
        if stats["avg_brier_score"] is not None:
            panel_text += (
                f"Avg Brier score: [cyan]{stats['avg_brier_score']:.4f}[/cyan] "
                f"(0.0=perfect, 0.25=random)\n"
                f"Win rate: [cyan]{stats['win_rate']:.1%}[/cyan]\n"
                f"Total P&L: [{'green' if stats['total_pnl_usdc'] >= 0 else 'red'}]"
                f"${stats['total_pnl_usdc']:.2f}[/]\n"
            )

        console.print(Panel(panel_text.strip(), title="Historical Performance", border_style="blue"))

    # ─── Shutdown ─────────────────────────────────────────────────────────────

    def _handle_shutdown(self, *_):
        logger.info("[Bot] Shutdown signal received")
        self._running = False
        if not config.bot.dry_run:
            self.executor.cancel_all_orders()
        self._print_stats()
        sys.exit(0)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit("/", 2)[0])  # Add project root to path

    bot = PolymarketBot()
    asyncio.run(bot.start())