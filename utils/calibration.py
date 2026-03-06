"""
calibration.py — tracks all predictions and resolved outcomes in SQLite.

Provides:
  - Brier score per market category and LLM provider
  - Calibration curve data (are our 70% calls right ~70% of the time?)
  - Per-provider performance for adaptive LLM weighting
  - P&L tracking
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from core.models import MarketOutcome, ResolvedTrade, TradeResult


DB_PATH = Path(__file__).parent.parent / "logs" / "calibration.db"


class CalibrationTracker:
    """
    Persists trade history and computes calibration metrics.
    Uses SQLite — no external dependencies, survives restarts.
    """

    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    # ─── Schema ───────────────────────────────────────────────────────────────

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id        TEXT UNIQUE,
                    market_question TEXT,
                    market_id       TEXT,
                    category        TEXT,
                    side            TEXT,         -- YES / NO
                    market_price    REAL,
                    our_probability REAL,
                    raw_edge        REAL,
                    size_usdc       REAL,
                    expected_value  REAL,
                    dry_run         INTEGER,
                    executed        INTEGER,
                    executed_at     TEXT,
                    -- Resolution fields (filled later)
                    resolved        INTEGER DEFAULT 0,
                    resolved_outcome TEXT,
                    pnl_usdc        REAL,
                    brier_score     REAL,
                    resolved_at     TEXT,
                    -- LLM breakdown (JSON)
                    llm_estimates   TEXT,
                    ensemble_prob   REAL,
                    confidence      REAL,
                    reasoning       TEXT
                );

                CREATE TABLE IF NOT EXISTS provider_stats (
                    provider        TEXT PRIMARY KEY,
                    total_trades    INTEGER DEFAULT 0,
                    sum_brier       REAL DEFAULT 0.0,
                    avg_brier       REAL DEFAULT 0.0,
                    last_updated    TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_trades_market
                    ON trades (market_id);
                CREATE INDEX IF NOT EXISTS idx_trades_resolved
                    ON trades (resolved);
            """)

    # ─── Recording ────────────────────────────────────────────────────────────

    def record_trade(self, result: TradeResult):
        """Save a trade to the database immediately after execution."""
        import json
        signal = result.signal

        # Serialise LLM estimates
        estimates_json = json.dumps([
            {
                "provider": e.provider,
                "model": e.model,
                "probability": e.probability,
                "confidence": e.confidence,
                "reasoning": e.reasoning,
            }
            for e in signal.ensemble.raw_estimates
        ])

        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trades (
                    order_id, market_question, market_id, category,
                    side, market_price, our_probability, raw_edge,
                    size_usdc, expected_value, dry_run, executed,
                    executed_at, llm_estimates, ensemble_prob,
                    confidence, reasoning
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.order_id,
                signal.market.question,
                signal.market.question_id,
                signal.market.category or "unknown",
                signal.side.value,
                signal.market_price,
                signal.estimated_probability,
                signal.raw_edge,
                signal.capped_size_usdc,
                signal.expected_value,
                int(result.dry_run),
                int(result.executed),
                result.timestamp.isoformat(),
                estimates_json,
                signal.ensemble.ensemble_probability,
                signal.ensemble.confidence,
                signal.ensemble.reasoning_summary[:500],
            ))

        logger.debug(f"[Calibration] Recorded trade: {result.order_id}")

    def resolve_trade(
        self,
        order_id: str,
        outcome: MarketOutcome,
        pnl_usdc: float,
    ) -> Optional[ResolvedTrade]:
        """
        Mark a trade as resolved. Computes Brier score and updates provider stats.
        Call this when a market settles.
        """
        # Fetch the original trade
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE order_id = ?", (order_id,)
            ).fetchone()

        if not row:
            logger.warning(f"[Calibration] Trade not found: {order_id}")
            return None

        # Brier score: (forecast - outcome)^2
        # outcome = 1.0 if our side won, 0.0 if it lost
        our_side_won = 1.0 if (
            (row["side"] == "YES" and outcome == MarketOutcome.YES) or
            (row["side"] == "NO" and outcome == MarketOutcome.NO)
        ) else 0.0

        brier = (row["our_probability"] - our_side_won) ** 2

        # Update trade record
        with self._conn() as conn:
            conn.execute("""
                UPDATE trades SET
                    resolved = 1,
                    resolved_outcome = ?,
                    pnl_usdc = ?,
                    brier_score = ?,
                    resolved_at = ?
                WHERE order_id = ?
            """, (
                outcome.value, pnl_usdc, brier,
                datetime.utcnow().isoformat(), order_id
            ))

        # Update provider stats
        self._update_provider_stats(order_id, brier)

        logger.info(
            f"[Calibration] Resolved {order_id} | outcome={outcome.value} | "
            f"pnl=${pnl_usdc:.2f} | brier={brier:.4f}"
        )
        return None  # Full ResolvedTrade object could be returned here if needed

    # ─── Analytics ────────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Return overall performance statistics."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM trades WHERE executed=1").fetchone()[0]
            resolved = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE resolved=1 AND executed=1"
            ).fetchone()[0]

            if resolved > 0:
                avg_brier = conn.execute(
                    "SELECT AVG(brier_score) FROM trades WHERE resolved=1"
                ).fetchone()[0]
                total_pnl = conn.execute(
                    "SELECT SUM(pnl_usdc) FROM trades WHERE resolved=1"
                ).fetchone()[0]
                win_rate = conn.execute(
                    "SELECT AVG(CASE WHEN pnl_usdc > 0 THEN 1.0 ELSE 0.0 END) "
                    "FROM trades WHERE resolved=1"
                ).fetchone()[0]
            else:
                avg_brier = total_pnl = win_rate = None

            dry_run_total = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE dry_run=1 AND executed=1"
            ).fetchone()[0]

        return {
            "total_trades": total,
            "resolved_trades": resolved,
            "dry_run_trades": dry_run_total,
            "avg_brier_score": round(avg_brier, 4) if avg_brier else None,
            "total_pnl_usdc": round(total_pnl, 2) if total_pnl else None,
            "win_rate": round(win_rate, 3) if win_rate else None,
            "note": "Brier score: 0.0=perfect, 0.25=random, 0.5=always wrong",
        }

    def get_calibration_curve(self) -> list[dict]:
        """
        Returns data for a calibration curve:
        Bucket our probability estimates into deciles and check actual hit rate.
        Well-calibrated: our 60-70% bucket should resolve YES ~65% of the time.
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT our_probability, resolved_outcome, side
                FROM trades
                WHERE resolved=1
            """).fetchall()

        if not rows:
            return []

        # Bucket into deciles
        buckets: dict[str, list[float]] = {f"{i*10}-{(i+1)*10}%": [] for i in range(10)}

        for row in rows:
            prob = row["our_probability"]
            won = 1.0 if (
                (row["side"] == "YES" and row["resolved_outcome"] == "YES") or
                (row["side"] == "NO" and row["resolved_outcome"] == "NO")
            ) else 0.0

            bucket_idx = min(int(prob * 10), 9)
            bucket_key = f"{bucket_idx*10}-{(bucket_idx+1)*10}%"
            buckets[bucket_key].append(won)

        curve = []
        for label, outcomes in buckets.items():
            if outcomes:
                curve.append({
                    "probability_bucket": label,
                    "actual_win_rate": round(sum(outcomes) / len(outcomes), 3),
                    "sample_count": len(outcomes),
                })

        return curve

    def get_provider_stats(self) -> list[dict]:
        """Return per-LLM-provider performance stats."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM provider_stats").fetchall()
        return [dict(r) for r in rows]

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _update_provider_stats(self, order_id: str, brier: float):
        """Update rolling per-provider Brier scores."""
        import json
        with self._conn() as conn:
            row = conn.execute(
                "SELECT llm_estimates FROM trades WHERE order_id=?", (order_id,)
            ).fetchone()

            if not row:
                return

            estimates = json.loads(row["llm_estimates"] or "[]")
            for est in estimates:
                provider = est.get("provider", "unknown")
                conn.execute("""
                    INSERT INTO provider_stats (provider, total_trades, sum_brier, avg_brier, last_updated)
                    VALUES (?, 1, ?, ?, ?)
                    ON CONFLICT(provider) DO UPDATE SET
                        total_trades = total_trades + 1,
                        sum_brier = sum_brier + excluded.sum_brier,
                        avg_brier = sum_brier / total_trades,
                        last_updated = excluded.last_updated
                """, (
                    provider, brier, brier, datetime.utcnow().isoformat()
                ))

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn