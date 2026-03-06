"""
analyse.py — one-shot market analyser for testing/debugging.

Usage:
  python analyse.py                      # Scan and show top 10 signals
  python analyse.py --new-only           # Only new markets
  python analyse.py --query "bitcoin"    # Search specific topic
  python analyse.py --stats              # Show calibration stats
  python analyse.py --curve             # Show calibration curve

This is the tool you run to verify the bot is finding real edges
before enabling live trading.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from rich.console import Console
from rich.table import Table

from core.config import config
from data.scanner import MarketScanner
from data.news_enricher import NewsEnricher
from strategies.tiered_forecaster import TieredForecaster
from strategies.edge_calculator import EdgeCalculator
from utils.calibration import CalibrationTracker

console = Console()


async def analyse(args):
    async with MarketScanner() as scanner, NewsEnricher() as enricher:
        forecaster = TieredForecaster()
        edge_calc = EdgeCalculator()
        calibration = CalibrationTracker()

        if args.stats:
            stats = calibration.get_summary()
            console.print(stats)
            curve = calibration.get_calibration_curve()
            if curve:
                t = Table(title="Calibration Curve")
                t.add_column("Bucket")
                t.add_column("Actual Win Rate")
                t.add_column("Samples")
                for row in curve:
                    t.add_row(row["probability_bucket"], f"{row['actual_win_rate']:.1%}", str(row["sample_count"]))
                console.print(t)
            return

        # Fetch markets
        if args.new_only:
            markets = await scanner.get_new_markets()
            console.print(f"[cyan]Found {len(markets)} new markets[/cyan]")
        else:
            markets = await scanner.get_active_markets(
                min_volume_24h=500.0,
                min_liquidity=config.bot.min_liquidity_usdc,
                limit=args.limit,
            )
            if args.query:
                q = args.query.lower()
                markets = [m for m in markets if q in m.question.lower() or q in (m.category or "").lower()]
            console.print(f"[cyan]Scanning {len(markets)} markets...[/cyan]")

        if not markets:
            console.print("[yellow]No markets found matching criteria[/yellow]")
            return

        # Enrich order books
        markets = await scanner.enrich_with_order_books(markets)

        # Process
        signals = []
        sem = asyncio.Semaphore(3)

        async def process_one(market):
            async with sem:
                try:
                    enriched = await enricher.enrich(market)
                    ensemble = await forecaster.estimate(enriched)
                    if ensemble:
                        signal = edge_calc.evaluate(enriched, ensemble)
                        if signal:
                            signals.append(signal)
                        else:
                            # Show all evaluations even without edge (verbose mode)
                            if args.verbose:
                                console.print(
                                    f"[dim]No edge: {market.question[:70]} | "
                                    f"LLM={ensemble.final_probability:.1%} "
                                    f"market={market.yes_price:.1%}[/dim]"
                                )
                except Exception as e:
                    logger.warning(f"Error: {e}")

        await asyncio.gather(*[process_one(m) for m in markets])

        # Display results
        ranked = EdgeCalculator.rank_signals(signals)
        console.print(f"\n[bold green]Found {len(ranked)} signals[/bold green]\n")

        t = Table(title="Signals Ranked by Expected Value", border_style="green")
        t.add_column("#")
        t.add_column("Market", max_width=50, overflow="fold")
        t.add_column("Category")
        t.add_column("Side")
        t.add_column("Market P", justify="right")
        t.add_column("Our P", justify="right")
        t.add_column("Edge", justify="right")
        t.add_column("Size $", justify="right")
        t.add_column("EV $", justify="right")
        t.add_column("Conf", justify="right")
        t.add_column("New")

        for i, s in enumerate(ranked[:20], 1):
            side_color = "green" if s.side.value == "YES" else "red"
            t.add_row(
                str(i),
                s.market.question[:50],
                s.market.category or "—",
                f"[{side_color}]{s.side.value}[/{side_color}]",
                f"{s.market_price:.1%}",
                f"{s.estimated_probability:.1%}",
                f"[green]{s.edge_after_fees:.1%}[/green]",
                f"${s.capped_size_usdc:.2f}",
                f"${s.expected_value:.2f}",
                f"{s.ensemble.confidence:.0%}",
                "🆕" if s.market.is_new else "",
            )
        console.print(t)

        if ranked and args.reasoning:
            console.print("\n[bold]Top signal reasoning:[/bold]")
            console.print(ranked[0].rationale)

        costs = forecaster.get_cost_summary()
        if costs["total_usd"] > 0:
            console.print(f"\n[dim]LLM cost this run: ${costs['total_usd']:.4f} | {costs['tip']}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket AI market analyser")
    parser.add_argument("--new-only", action="store_true", help="Only scan new markets")
    parser.add_argument("--query", type=str, help="Filter markets by keyword")
    parser.add_argument("--limit", type=int, default=100, help="Max markets to scan")
    parser.add_argument("--stats", action="store_true", help="Show calibration stats")
    parser.add_argument("--verbose", action="store_true", help="Show all evaluations")
    parser.add_argument("--reasoning", action="store_true", help="Show LLM reasoning for top signal")
    args = parser.parse_args()

    asyncio.run(analyse(args))