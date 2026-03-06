"""SQLite persistence for trades, outcomes, and ML training data."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from core.models import Direction, PerformanceSnapshot, Position, SignalSource, TradeStatus

DB_PATH = Path(__file__).parent.parent / "logs" / "trades.db"


class TradeStore:
    """Records all trades and computes performance metrics."""

    def __init__(self, db_path: Path = DB_PATH, suffix: str = ""):
        if suffix:
            db_path = db_path.with_stem(db_path.stem + suffix)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id        TEXT UNIQUE,
                    instrument      TEXT,
                    direction       TEXT,
                    strategy        TEXT,
                    entry_price     REAL,
                    exit_price      REAL,
                    stop_loss       REAL,
                    take_profit     REAL,
                    size_units      REAL,
                    risk_amount     REAL,
                    pnl             REAL,
                    pnl_pips        REAL,
                    status          TEXT DEFAULT 'open',
                    exit_reason     TEXT,
                    regime          TEXT,
                    signal_strength REAL,
                    signal_confidence REAL,
                    entry_features  TEXT,
                    opened_at       TEXT,
                    closed_at       TEXT,
                    dry_run         INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS strategy_performance (
                    strategy        TEXT,
                    regime          TEXT,
                    window          TEXT,
                    total_trades    INTEGER DEFAULT 0,
                    wins            INTEGER DEFAULT 0,
                    losses          INTEGER DEFAULT 0,
                    total_pnl       REAL DEFAULT 0.0,
                    avg_win         REAL DEFAULT 0.0,
                    avg_loss        REAL DEFAULT 0.0,
                    updated_at      TEXT,
                    PRIMARY KEY (strategy, regime, window)
                );

                CREATE INDEX IF NOT EXISTS idx_trades_instrument
                    ON trades (instrument);
                CREATE INDEX IF NOT EXISTS idx_trades_status
                    ON trades (status);
                CREATE INDEX IF NOT EXISTS idx_trades_strategy
                    ON trades (strategy);
            """)

    # ── Record / Update ──────────────────────────────────────────────────────

    def record_trade(self, position: Position, dry_run: bool = True):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trades (
                    trade_id, instrument, direction, strategy,
                    entry_price, stop_loss, take_profit, size_units,
                    risk_amount, status, regime, signal_strength,
                    signal_confidence, entry_features, opened_at, dry_run
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.trade_id,
                position.instrument,
                position.direction.value,
                position.signal.source.value,
                position.entry_price,
                position.stop_loss,
                position.take_profit,
                position.size_units,
                position.risk_usdc,
                position.status.value,
                position.signal.regime.value,
                position.signal.strength,
                position.signal.confidence,
                json.dumps(position.entry_features),
                position.opened_at.isoformat(),
                int(dry_run),
            ))

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        pnl_pips: float,
        exit_reason: str,
    ):
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                UPDATE trades SET
                    exit_price = ?, pnl = ?, pnl_pips = ?,
                    status = ?, exit_reason = ?, closed_at = ?
                WHERE trade_id = ?
            """, (exit_price, pnl, pnl_pips, "closed", exit_reason, now, trade_id))

        self._update_strategy_performance(trade_id)

    # ── Queries ──────────────────────────────────────────────────────────────

    def get_closed_trades(
        self,
        strategy: Optional[str] = None,
        instrument: Optional[str] = None,
        regime: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict]:
        query = "SELECT * FROM trades WHERE status='closed'"
        params: list = []
        if strategy:
            query += " AND strategy=?"
            params.append(strategy)
        if instrument:
            query += " AND instrument=?"
            params.append(instrument)
        if regime:
            query += " AND regime=?"
            params.append(regime)
        query += " ORDER BY closed_at DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_open_trade_count(self, instrument: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) FROM trades WHERE status='open'"
        params: list = []
        if instrument:
            query += " AND instrument=?"
            params.append(instrument)
        with self._conn() as conn:
            return conn.execute(query, params).fetchone()[0]

    def get_open_trades(self) -> list[dict]:
        """Retrieve all open trades for position recovery after restart."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='open' ORDER BY opened_at ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_performance(
        self,
        strategy: Optional[str] = None,
        regime: Optional[str] = None,
        last_n: Optional[int] = None,
    ) -> PerformanceSnapshot:
        query = "SELECT pnl, pnl_pips FROM trades WHERE status='closed'"
        params: list = []
        if strategy:
            query += " AND strategy=?"
            params.append(strategy)
        if regime:
            query += " AND regime=?"
            params.append(regime)
        query += " ORDER BY closed_at DESC"
        if last_n:
            query += " LIMIT ?"
            params.append(last_n)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return PerformanceSnapshot()

        pnls = [r["pnl"] for r in rows if r["pnl"] is not None]
        if not pnls:
            return PerformanceSnapshot()

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl = sum(pnls)
        total_wins = sum(wins)
        total_losses = abs(sum(losses))

        # Max drawdown
        cumulative = np.cumsum(pnls[::-1])  # oldest first
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # Sharpe (annualised, assuming daily resolution)
        if len(pnls) > 1:
            ret_std = float(np.std(pnls, ddof=1))
            sharpe = (float(np.mean(pnls)) / ret_std * np.sqrt(252)) if ret_std > 0 else 0.0
        else:
            sharpe = 0.0

        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        win_rate = len(wins) / len(pnls) if pnls else 0.0

        return PerformanceSnapshot(
            total_trades=len(pnls),
            wins=len(wins),
            losses=len(losses),
            total_pnl=round(total_pnl, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            win_rate=round(win_rate, 4),
            profit_factor=round(total_wins / total_losses, 3) if total_losses > 0 else 0.0,
            sharpe_ratio=round(sharpe, 3),
            max_drawdown=round(max_dd, 2),
            expectancy=round(avg_win * win_rate + avg_loss * (1 - win_rate), 4),
        )

    def get_ml_training_data(self, min_trades: int = 50) -> Optional[tuple]:
        """
        Returns (X, y) for ML training where:
          X = entry_features (dict → array)
          y = 1 if trade was profitable, 0 otherwise
        """
        trades = self.get_closed_trades(limit=5000)
        if len(trades) < min_trades:
            return None

        rows = []
        labels = []
        for t in trades:
            features_str = t.get("entry_features")
            if not features_str:
                continue
            try:
                features = json.loads(features_str)
                features["strategy_" + t.get("strategy", "unknown")] = 1.0
                features["regime_" + t.get("regime", "unknown")] = 1.0
                features["signal_strength"] = t.get("signal_strength", 0.0)
                features["signal_confidence"] = t.get("signal_confidence", 0.0)
                rows.append(features)
                labels.append(1 if (t.get("pnl") or 0) > 0 else 0)
            except (json.JSONDecodeError, TypeError):
                continue

        if len(rows) < min_trades:
            return None

        # Align feature dicts to common columns
        all_keys = sorted(set().union(*[r.keys() for r in rows]))
        X = np.array([[r.get(k, 0.0) for k in all_keys] for r in rows])
        y = np.array(labels)
        return X, y, all_keys

    # ── Internal ─────────────────────────────────────────────────────────────

    def _update_strategy_performance(self, trade_id: str):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT strategy, regime, pnl FROM trades WHERE trade_id=?",
                (trade_id,)
            ).fetchone()

        if not row or row["pnl"] is None:
            return

        strategy = row["strategy"]
        regime = row["regime"] or "unknown"
        pnl = row["pnl"]
        won = pnl > 0

        for window in ["all", regime]:
            perf = self.get_performance(strategy=strategy, regime=window if window != "all" else None)
            with self._conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO strategy_performance
                        (strategy, regime, window, total_trades, wins, losses,
                         total_pnl, avg_win, avg_loss, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy, regime, window,
                    perf.total_trades, perf.wins, perf.losses,
                    perf.total_pnl, perf.avg_win, perf.avg_loss,
                    datetime.utcnow().isoformat(),
                ))

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn
