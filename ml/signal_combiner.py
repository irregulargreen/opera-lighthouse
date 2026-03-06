"""
Signal combiner — the meta-learner that makes the system compound.

Takes signals from all strategies and learns which ones to trust
in which conditions. This is where the system improves over time.

Cold start: equal-weight all strategies, regime-based adjustments
After 50+ trades: gradient boosting predicts P(profit) from signal features
After 200+ trades: per-regime, per-strategy adaptive weighting

Key insight: individual strategies have thin edges (~52-55% win rate).
But a model that learns WHEN each strategy works can push the combined
win rate to 55-60% by sitting out bad conditions for each strategy.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from core.models import Regime, Signal

MODEL_PATH = Path(__file__).parent.parent / "logs" / "signal_model.pkl"
DATA_PATH = Path(__file__).parent.parent / "logs" / "signal_data.json"
MIN_SAMPLES = 50
RETRAIN_EVERY = 25


# Default strategy weights per regime (before ML training)
DEFAULT_WEIGHTS = {
    Regime.TRENDING_UP: {"trend_following": 0.6, "mean_reversion": 0.1, "breakout": 0.3},
    Regime.TRENDING_DOWN: {"trend_following": 0.6, "mean_reversion": 0.1, "breakout": 0.3},
    Regime.RANGING: {"trend_following": 0.1, "mean_reversion": 0.6, "breakout": 0.3},
    Regime.VOLATILE: {"trend_following": 0.2, "mean_reversion": 0.2, "breakout": 0.6},
    Regime.UNKNOWN: {"trend_following": 0.4, "mean_reversion": 0.3, "breakout": 0.3},
}


class SignalCombiner:
    """
    Decides which signals to act on and how to weight them.

    Two modes:
      1. Heuristic: regime-based strategy weighting (cold start)
      2. ML: gradient boosting predicts P(profitable) from all features
         Then scales signal strength by predicted win probability

    The ML model trains on resolved trade outcomes, learning patterns like:
      - "Trend signals with ADX>35 and RSI<60 win 65% of the time"
      - "Breakout signals during low volatility with volume spike win 55%"
      - "Reversion signals when ADX is rising (hidden trend) lose 60%"
    """

    def __init__(self, suffix: str = ""):
        self._model = None
        self._is_fitted = False
        self._feature_names: list[str] = []
        self._training_data: list[dict] = []
        self._samples_since_refit = 0
        self._model_auc: float = 0.5
        self._strategy_win_rates: dict[str, dict] = {}
        self._suffix = suffix
        self._model_path = Path(str(MODEL_PATH).replace(".pkl", f"{suffix}.pkl")) if suffix else MODEL_PATH
        self._data_path = Path(str(DATA_PATH).replace(".json", f"{suffix}.json")) if suffix else DATA_PATH
        self._load()

    def rank_signals(
        self,
        signals: list[Signal],
        regime: Regime,
    ) -> list[tuple[Signal, float]]:
        """
        Rank and score signals. Returns [(signal, combined_score)] sorted best-first.
        Score incorporates strategy weight, signal strength, and ML prediction.
        """
        scored: list[tuple[Signal, float]] = []

        for signal in signals:
            if signal.direction.value == "flat":
                continue

            # Base: strategy weight for this regime × signal strength
            regime_weights = DEFAULT_WEIGHTS.get(regime, DEFAULT_WEIGHTS[Regime.UNKNOWN])
            strategy_weight = regime_weights.get(signal.source.value, 0.33)
            base_score = signal.strength * strategy_weight

            # ML adjustment: if model is trained, adjust based on P(profitable)
            ml_multiplier = 1.0
            if self._is_fitted:
                win_prob = self._predict_win_prob(signal)
                # Scale: 0.4 win prob → 0.6x, 0.5 → 1.0x, 0.65 → 1.3x
                ml_multiplier = 0.6 + (win_prob - 0.4) * 2.8
                ml_multiplier = max(0.3, min(1.5, ml_multiplier))

            combined = round(base_score * ml_multiplier, 4)
            scored.append((signal, combined))

        return sorted(scored, key=lambda x: x[1], reverse=True)

    def should_trade(self, signal: Signal, score: float) -> tuple[bool, str]:
        """Final go/no-go decision on a signal."""
        from core.config import config
        min_strength = config.trading.min_signal_strength

        if score < min_strength:
            return False, f"score {score:.3f} < min {min_strength}"

        if self._is_fitted:
            win_prob = self._predict_win_prob(signal)
            if win_prob < 0.48:
                return False, f"ML win_prob {win_prob:.1%} too low"
            return True, f"ML approved: win_prob={win_prob:.1%} score={score:.3f}"

        return True, f"heuristic: score={score:.3f}"

    def record_outcome(self, signal: Signal, won: bool, pnl: float, regime: Regime):
        """Record whether a trade was profitable (for ML training)."""
        features = self._signal_to_features(signal, regime)
        self._training_data.append({
            "features": features,
            "won": int(won),
            "pnl": pnl,
            "strategy": signal.source.value,
            "regime": regime.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._samples_since_refit += 1

        # Track per-strategy, per-regime win rates
        key = f"{signal.source.value}:{regime.value}"
        if key not in self._strategy_win_rates:
            self._strategy_win_rates[key] = {"wins": 0, "total": 0}
        self._strategy_win_rates[key]["total"] += 1
        if won:
            self._strategy_win_rates[key]["wins"] += 1

        if (
            len(self._training_data) >= MIN_SAMPLES
            and self._samples_since_refit >= RETRAIN_EVERY
        ):
            self.refit()

        self._save_data()

    def refit(self) -> bool:
        n = len(self._training_data)
        if n < MIN_SAMPLES or not (SKLEARN_AVAILABLE or LGBM_AVAILABLE):
            return False

        X = np.array([
            [s["features"].get(k, 0.0) for k in self._get_feature_keys()]
            for s in self._training_data
        ])
        y = np.array([s["won"] for s in self._training_data])
        self._feature_names = self._get_feature_keys()

        if len(np.unique(y)) < 2:
            return False

        if LGBM_AVAILABLE and n >= 200:
            model = lgb.LGBMClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                min_child_samples=max(5, n // 50),
                class_weight="balanced", random_state=42, verbose=-1,
            )
        elif SKLEARN_AVAILABLE:
            model = GradientBoostingClassifier(
                n_estimators=80, max_depth=4,
                min_samples_leaf=max(3, n // 50),
                random_state=42,
            )
        else:
            return False

        cv = min(5, n // 20)
        if cv >= 2:
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
                self._model_auc = float(scores.mean())
                logger.info(
                    f"[Signal] CV AUC: {self._model_auc:.3f} ± {scores.std():.3f} ({n} samples)"
                )
            except Exception:
                pass

        model.fit(X, y)
        self._model = model
        self._is_fitted = True
        self._samples_since_refit = 0

        self._log_importances()
        self._save_model()
        return True

    def get_insights(self) -> dict:
        """What has the combiner learned?"""
        insights: dict = {
            "is_fitted": self._is_fitted,
            "training_samples": len(self._training_data),
            "model_auc": round(self._model_auc, 4),
        }

        # Per-strategy, per-regime win rates
        win_rates = {}
        for key, stats in self._strategy_win_rates.items():
            if stats["total"] >= 5:
                win_rates[key] = {
                    "win_rate": round(stats["wins"] / stats["total"], 3),
                    "trades": stats["total"],
                }
        insights["strategy_regime_win_rates"] = win_rates

        # Feature importances
        if self._is_fitted and hasattr(self._model, "feature_importances_"):
            imp = dict(sorted(
                zip(self._feature_names, self._model.feature_importances_),
                key=lambda x: x[1], reverse=True,
            )[:10])
            insights["top_features"] = {k: round(float(v), 4) for k, v in imp.items()}

        return insights

    # ── Internal ─────────────────────────────────────────────────────────────

    def _predict_win_prob(self, signal: Signal) -> float:
        if not self._is_fitted or self._model is None:
            return 0.5
        try:
            features = self._signal_to_features(signal, signal.regime)
            X = np.array([[features.get(k, 0.0) for k in self._feature_names]])
            return float(self._model.predict_proba(X)[0][1])
        except Exception:
            return 0.5

    def _signal_to_features(self, signal: Signal, regime: Regime) -> dict[str, float]:
        """Combine signal features with meta-features for the ML model."""
        f = dict(signal.features)
        f["signal_strength"] = signal.strength
        f["signal_confidence"] = signal.confidence
        f["is_long"] = float(signal.direction.value == "long")
        # Strategy one-hot
        f["strat_trend"] = float(signal.source.value == "trend_following")
        f["strat_reversion"] = float(signal.source.value == "mean_reversion")
        f["strat_breakout"] = float(signal.source.value == "breakout")
        # Regime one-hot
        f["regime_trend_up"] = float(regime == Regime.TRENDING_UP)
        f["regime_trend_down"] = float(regime == Regime.TRENDING_DOWN)
        f["regime_ranging"] = float(regime == Regime.RANGING)
        f["regime_volatile"] = float(regime == Regime.VOLATILE)
        # Risk/reward ratio
        if signal.stop_loss and signal.entry_price:
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.entry_price) if signal.take_profit else risk * 2
            f["rr_ratio"] = reward / risk if risk > 0 else 0
        return f

    def _get_feature_keys(self) -> list[str]:
        if self._training_data:
            return sorted(self._training_data[0]["features"].keys())
        return []

    def _log_importances(self):
        if self._is_fitted and hasattr(self._model, "feature_importances_"):
            imp = sorted(
                zip(self._feature_names, self._model.feature_importances_),
                key=lambda x: x[1], reverse=True,
            )[:5]
            logger.info(f"[Signal] Top features: {[(k, round(float(v), 3)) for k, v in imp]}")

    def _save_data(self):
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        self._data_path.write_text(json.dumps({
            "training_data": self._training_data[-5000:],
            "strategy_win_rates": self._strategy_win_rates,
        }, indent=2))

    def _save_model(self):
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "feature_names": self._feature_names,
                "is_fitted": self._is_fitted,
                "model_auc": self._model_auc,
            }, f)

    def _load(self):
        if self._data_path.exists():
            try:
                p = json.loads(self._data_path.read_text())
                self._training_data = p.get("training_data", [])
                self._strategy_win_rates = p.get("strategy_win_rates", {})
                logger.info(f"[Signal] Loaded {len(self._training_data)} training samples")
            except Exception:
                pass
        if self._model_path.exists():
            try:
                with open(self._model_path, "rb") as f:
                    p = pickle.load(f)
                self._model = p["model"]
                self._feature_names = p["feature_names"]
                self._is_fitted = p["is_fitted"]
                self._model_auc = p.get("model_auc", 0.5)
                logger.info(f"[Signal] Loaded model (AUC={self._model_auc:.3f})")
            except Exception:
                pass
