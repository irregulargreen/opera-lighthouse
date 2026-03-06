"""
meta_learner.py — the compound learning layer.

The fundamental insight: the LLM's probability estimate is just ONE signal.
A meta-learner trained on resolved trades learns to combine:

  - LLM probability estimate (and gap from crowd)
  - LLM confidence + inter-model consensus
  - News signal strength (how much news, how recent, source quality)
  - Market features (liquidity, spread, time-to-close, category)
  - Crowd price dynamics (is price moving toward or away from LLM estimate?)
  - Whether the market is new (first-mover window)
  - Historical accuracy for this category + this price range

Output: P(our_trade_wins | all_signals)

This replaces the hand-tuned edge threshold with a data-driven one.
Over time it learns things like:
  - "LLM confidence 70%+ on political markets with >3 news articles 
     and spread >4¢ has 71% win rate historically — take the trade"
  - "On crypto markets where LLM and crowd differ by <6%, we're 
     not better than random — skip even if 'edge' looks positive"
  - "New markets in the 30-70% price range with high volume velocity 
     are our single best category — double Kelly sizing"

Architecture:
  Online learning: updates incrementally after every resolved trade
  Model: gradient boosting (LightGBM) → retrained every 50 resolved trades
  Cold start: heuristic scoring until 100 resolved trades accumulated
  Feature importance: tracked to understand what's actually driving edge
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
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score, brier_score_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from core.models import EnrichedMarket, EnsembleEstimate, TradeSignal


META_MODEL_PATH = Path(__file__).parent.parent / "logs" / "meta_learner.pkl"
META_DATA_PATH = Path(__file__).parent.parent / "logs" / "meta_training_data.json"

# Minimum resolved trades before switching from heuristic to model
MIN_SAMPLES_COLD_START = 100
# Retrain model every N new resolved trades
RETRAIN_EVERY_N = 25


class MetaLearner:
    """
    The compound learning layer.

    Takes the full signal stack (LLM estimate + market features + news signal)
    and predicts actual win probability based on historical resolved trades.

    This is what makes the system genuinely self-improving — every resolved
    trade makes future predictions sharper.
    """

    def __init__(self):
        self._model = None
        self._is_fitted = False
        self._training_data: list[dict] = []
        self._trades_since_last_fit = 0
        self._feature_names: list[str] = []

        # Performance tracking
        self._model_auc: float = 0.0
        self._model_brier: float = 0.25
        self._baseline_win_rate: float = 0.5

        self._load_from_disk()

    # ─── Public interface ─────────────────────────────────────────────────────

    def predict_win_probability(
        self,
        signal: TradeSignal,
        enriched: EnrichedMarket,
    ) -> tuple[float, float]:
        """
        Predict the true probability our trade wins, given all signals.

        Returns: (win_probability, confidence_in_prediction)
          win_probability: 0.0-1.0, estimated P(trade resolves in our favour)
          confidence: 0.0-1.0, how much to trust this prediction
                      (low early on, increases as more data accumulates)
        """
        features = self._extract_meta_features(signal, enriched)

        if self._is_fitted and self._model is not None:
            X = np.array([list(features.values())])
            try:
                win_prob = float(self._model.predict_proba(X)[0][1])
                # Confidence scales with number of training samples and model AUC
                confidence = self._prediction_confidence(features)
                return round(win_prob, 4), round(confidence, 4)
            except Exception as e:
                logger.debug(f"[Meta] Model predict failed: {e}")

        # Cold start: use heuristic estimate
        return self._heuristic_win_prob(signal), 0.3

    def should_trade(
        self,
        signal: TradeSignal,
        enriched: EnrichedMarket,
        min_win_prob: float = 0.55,
    ) -> tuple[bool, str]:
        """
        Final gate: should we place this trade?

        Combines the meta-learner's win probability estimate with the
        raw edge signal. More conservative than raw edge alone.

        Returns: (should_trade, reason)
        """
        win_prob, confidence = self.predict_win_probability(signal, enriched)

        if not self._is_fitted:
            # Pre-training: trust the raw edge signal with standard threshold
            reason = f"[cold-start] raw edge={signal.edge_after_fees:.1%} — using edge threshold"
            return signal.edge_after_fees >= 0.05, reason

        # With fitted model: use meta-learner's estimate
        # Scale required win rate by confidence in the prediction
        # Low confidence → require higher win prob before trading
        adjusted_threshold = min_win_prob + (1 - confidence) * 0.10

        should = win_prob >= adjusted_threshold
        reason = (
            f"meta_win_prob={win_prob:.1%} "
            f"threshold={adjusted_threshold:.1%} "
            f"confidence={confidence:.0%} "
            f"model_auc={self._model_auc:.3f}"
        )
        return should, reason

    def adjusted_kelly_multiplier(
        self,
        signal: TradeSignal,
        enriched: EnrichedMarket,
    ) -> float:
        """
        Returns a multiplier for Kelly sizing based on meta-learner confidence.

        High confidence + high win prob → full Kelly multiplier (1.0)
        Low confidence or below-baseline win prob → reduce sizing
        High win prob in well-fitted model → allow up to 1.2× (max)

        This replaces the fixed KELLY_FRACTION with an adaptive version.
        """
        if not self._is_fitted:
            return 1.0  # No adjustment pre-training

        win_prob, confidence = self.predict_win_probability(signal, enriched)

        # Base: scale by win probability vs baseline
        win_ratio = win_prob / max(self._baseline_win_rate, 0.5)

        # Dampen by confidence in the meta-prediction
        multiplier = 0.5 + 0.5 * confidence * win_ratio

        # Cap between 0.3 (minimum) and 1.3 (maximum — don't bet too big on any model)
        return round(max(0.3, min(1.3, multiplier)), 3)

    def record_outcome(
        self,
        signal: TradeSignal,
        enriched: EnrichedMarket,
        won: bool,
        pnl: float,
    ):
        """
        Record a resolved trade. This is the core learning signal.
        Call after every market resolution.
        """
        features = self._extract_meta_features(signal, enriched)

        sample = {
            "features": features,
            "won": int(won),
            "pnl": pnl,
            "market_id": signal.market.question_id,
            "category": signal.market.category or "unknown",
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            # Store the signals that were used, for analysis
            "llm_prob": signal.estimated_probability,
            "market_price": signal.market_price,
            "raw_edge": signal.raw_edge,
            "edge_after_fees": signal.edge_after_fees,
        }
        self._training_data.append(sample)
        self._trades_since_last_fit += 1

        logger.info(
            f"[Meta] Recorded: {'✅ WIN' if won else '❌ LOSS'} | "
            f"pnl=${pnl:.2f} | total_samples={len(self._training_data)}"
        )

        # Retrain periodically
        if (
            len(self._training_data) >= MIN_SAMPLES_COLD_START
            and self._trades_since_last_fit >= RETRAIN_EVERY_N
        ):
            self.refit()

        self._save_data()

    def refit(self) -> bool:
        """Retrain the meta-learner on all accumulated data."""
        n = len(self._training_data)
        if n < MIN_SAMPLES_COLD_START:
            logger.info(f"[Meta] Not enough data ({n}/{MIN_SAMPLES_COLD_START})")
            return False

        if not (SKLEARN_AVAILABLE or LGBM_AVAILABLE):
            logger.warning("[Meta] No ML libraries available")
            return False

        X = np.array([list(s["features"].values()) for s in self._training_data])
        y = np.array([s["won"] for s in self._training_data])
        self._feature_names = list(self._training_data[0]["features"].keys())
        self._baseline_win_rate = float(y.mean())

        logger.info(f"[Meta] Retraining on {n} samples | baseline win rate: {self._baseline_win_rate:.1%}")

        # Choose model
        if LGBM_AVAILABLE and n >= 300:
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                min_child_samples=max(5, n // 50),
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            )
        elif SKLEARN_AVAILABLE:
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=max(3, n // 50),
                random_state=42,
            )
        else:
            return False

        # Cross-validation
        cv_folds = min(5, n // 20)
        if cv_folds >= 2:
            auc_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="roc_auc")
            brier_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="neg_brier_score")
            self._model_auc = float(auc_scores.mean())
            self._model_brier = float(-brier_scores.mean())
            logger.info(
                f"[Meta] CV results: AUC={self._model_auc:.3f}±{auc_scores.std():.3f} | "
                f"Brier={self._model_brier:.4f}"
            )

        model.fit(X, y)
        self._model = model
        self._is_fitted = True
        self._trades_since_last_fit = 0

        self._log_feature_importances()
        self._save_model()
        return True

    def get_insights(self) -> dict:
        """
        Return human-readable insights about what the meta-learner has learned.
        Useful for tuning strategy and understanding where edge is strongest.
        """
        if not self._training_data:
            return {"status": "no_data"}

        # Win rate by category
        by_cat: dict[str, list] = {}
        for s in self._training_data:
            cat = s.get("category", "unknown")
            by_cat.setdefault(cat, []).append(s["won"])

        category_win_rates = {
            cat: {
                "win_rate": round(float(np.mean(outcomes)), 3),
                "samples": len(outcomes),
            }
            for cat, outcomes in by_cat.items()
            if len(outcomes) >= 5
        }

        # Win rate by edge bucket
        edge_buckets: dict[str, list] = {}
        for s in self._training_data:
            edge = s.get("edge_after_fees", 0)
            bucket = f"{int(edge * 100 // 2) * 2}-{int(edge * 100 // 2) * 2 + 2}%"
            edge_buckets.setdefault(bucket, []).append(s["won"])

        edge_win_rates = {
            bucket: {
                "win_rate": round(float(np.mean(v)), 3),
                "samples": len(v),
            }
            for bucket, v in edge_buckets.items()
            if len(v) >= 5
        }

        # Feature importances
        importances = {}
        if self._is_fitted and self._model is not None:
            if hasattr(self._model, "feature_importances_"):
                imp = self._model.feature_importances_
                importances = dict(
                    sorted(
                        zip(self._feature_names, imp),
                        key=lambda x: x[1], reverse=True
                    )[:10]
                )

        return {
            "status": "fitted" if self._is_fitted else "heuristic",
            "total_resolved_trades": len(self._training_data),
            "overall_win_rate": round(self._baseline_win_rate, 3),
            "model_auc": round(self._model_auc, 4),
            "model_brier": round(self._model_brier, 4),
            "win_rate_by_category": category_win_rates,
            "win_rate_by_edge_bucket": edge_win_rates,
            "top_predictive_features": importances,
            "interpretation": self._interpret(),
        }

    # ─── Feature engineering ──────────────────────────────────────────────────

    def _extract_meta_features(
        self, signal: TradeSignal, enriched: EnrichedMarket
    ) -> dict[str, float]:
        """
        Extract features combining all signal layers for the meta-learner.
        These are the inputs to "should I trust this trade?"
        """
        market = enriched.market
        ensemble = signal.ensemble

        # Time to close
        if market.close_time:
            days_to_close = max(0, (market.close_time - datetime.now(timezone.utc)).days)
        else:
            days_to_close = 30.0

        # News signal strength
        n_articles = len(enriched.news_items)
        avg_news_relevance = (
            float(np.mean([i.relevance_score for i in enriched.news_items]))
            if enriched.news_items else 0.0
        )
        # Recency: what fraction of news is from last 48h?
        recent_count = sum(
            1 for item in enriched.news_items
            if item.published_at and
            (datetime.now(timezone.utc) - item.published_at).total_seconds() < 172800
        )
        news_recency = recent_count / max(n_articles, 1)

        # LLM signal features
        llm_prob = signal.estimated_probability
        market_price = signal.market_price
        raw_edge = signal.raw_edge
        llm_confidence = ensemble.confidence
        llm_consensus = ensemble.consensus  # std dev (lower = models agree more)
        n_models_used = len(ensemble.raw_estimates)

        # Per-model agreement signal
        if ensemble.raw_estimates:
            model_probs = [e.probability for e in ensemble.raw_estimates]
            model_max_diff = max(model_probs) - min(model_probs)
        else:
            model_max_diff = 0.0

        # Market features
        liq = max(1.0, market.liquidity)
        spread = market.spread
        vol_velocity = (
            market.volume_24h / max(market.volume_total, 1.0)
        )

        # Category encoding
        cat = (market.category or "other").lower()
        cat_is_politics = float("polit" in cat or "elect" in cat)
        cat_is_crypto = float("crypto" in cat or "btc" in cat or "eth" in cat)
        cat_is_sports = float("sport" in cat or "nba" in cat or "nfl" in cat)
        cat_is_finance = float("financ" in cat or "econ" in cat or "market" in cat)

        return {
            # Core signal
            "llm_prob": llm_prob,
            "market_price": market_price,
            "raw_edge": raw_edge,
            "edge_after_fees": signal.edge_after_fees,
            "price_distance_from_half": abs(market_price - 0.5),

            # LLM quality signals
            "llm_confidence": llm_confidence,
            "llm_consensus": llm_consensus,          # LOW = models agree = more reliable
            "model_max_disagreement": model_max_diff,
            "n_models_used": float(n_models_used),

            # News signals
            "n_news_articles": float(min(n_articles, 20)),
            "avg_news_relevance": avg_news_relevance,
            "news_recency": news_recency,
            "has_news": float(n_articles > 0),

            # Market quality signals
            "log_liquidity": float(np.log1p(liq)),
            "spread": float(min(spread, 0.5)),
            "volume_velocity": float(vol_velocity),
            "is_new_market": float(market.is_new),
            "days_to_close": float(min(days_to_close, 365)),
            "is_short_horizon": float(days_to_close < 7),

            # Category
            "cat_politics": cat_is_politics,
            "cat_crypto": cat_is_crypto,
            "cat_sports": cat_is_sports,
            "cat_finance": cat_is_finance,

            # Interaction features (these often matter most)
            "edge_x_confidence": signal.edge_after_fees * llm_confidence,
            "edge_x_news": signal.edge_after_fees * min(n_articles / 5.0, 1.0),
            "confidence_x_consensus": llm_confidence * (1 - llm_consensus),
            "new_market_x_edge": float(market.is_new) * signal.edge_after_fees,
            "liquidity_x_spread": np.log1p(liq) * min(spread, 0.5),
        }

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _prediction_confidence(self, features: dict) -> float:
        """
        How confident should we be in the meta-learner's prediction?
        Scales with: training set size, model AUC, and feature similarity to training data.
        """
        # Base confidence from training size (asymptotes at 1.0)
        size_confidence = min(1.0, len(self._training_data) / 500.0)

        # AUC contribution (0.5 AUC = random, 1.0 AUC = perfect)
        auc_confidence = max(0.0, (self._model_auc - 0.5) * 2.0)

        # Combined (geometric mean to be conservative)
        return float(np.sqrt(size_confidence * auc_confidence))

    def _heuristic_win_prob(self, signal: TradeSignal) -> float:
        """Cold-start heuristic: maps edge + confidence to rough win probability."""
        edge = signal.edge_after_fees
        conf = signal.ensemble.confidence
        consensus = signal.ensemble.consensus

        # Base: slightly above random for any positive edge
        base = 0.50 + edge * 0.8

        # Confidence adjustment
        base *= (0.8 + 0.2 * conf)

        # Consensus penalty (high std dev = models disagree = less reliable)
        base *= (1.0 - consensus * 0.3)

        # New market bonus
        if signal.market.is_new:
            base *= 1.05

        return round(max(0.45, min(0.85, base)), 4)

    def _interpret(self) -> str:
        """Generate a plain-English interpretation of what the model has learned."""
        if not self._is_fitted:
            return "Model not yet fitted — accumulating training data"

        lines = []
        if self._model_auc > 0.65:
            lines.append(f"Strong signal (AUC={self._model_auc:.3f}): meta-learner is finding real patterns")
        elif self._model_auc > 0.55:
            lines.append(f"Moderate signal (AUC={self._model_auc:.3f}): some patterns emerging")
        else:
            lines.append(f"Weak signal (AUC={self._model_auc:.3f}): more data needed or edge is noisy")

        imp = self.get_feature_importances_dict()
        if imp:
            top = list(imp.items())[:3]
            lines.append(f"Most predictive: {', '.join(k for k, _ in top)}")

        return " | ".join(lines)

    def get_feature_importances_dict(self) -> dict:
        if not self._is_fitted or self._model is None:
            return {}
        if hasattr(self._model, "feature_importances_"):
            imp = self._model.feature_importances_
            return dict(sorted(
                zip(self._feature_names, [round(float(i), 4) for i in imp]),
                key=lambda x: x[1], reverse=True
            ))
        return {}

    def _log_feature_importances(self):
        imp = self.get_feature_importances_dict()
        if imp:
            top5 = list(imp.items())[:5]
            logger.info(f"[Meta] Top predictive features: {top5}")

    # ─── Persistence ──────────────────────────────────────────────────────────

    def _save_data(self):
        META_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        META_DATA_PATH.write_text(
            json.dumps(self._training_data[-10_000:], indent=2)
        )

    def _save_model(self):
        META_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(META_MODEL_PATH, "wb") as f:
            pickle.dump({
                "model": self._model,
                "feature_names": self._feature_names,
                "is_fitted": self._is_fitted,
                "model_auc": self._model_auc,
                "model_brier": self._model_brier,
                "baseline_win_rate": self._baseline_win_rate,
            }, f)

    def _load_from_disk(self):
        # Load training data
        if META_DATA_PATH.exists():
            try:
                self._training_data = json.loads(META_DATA_PATH.read_text())
                logger.info(f"[Meta] Loaded {len(self._training_data)} training samples")
            except Exception as e:
                logger.warning(f"[Meta] Failed to load training data: {e}")

        # Load fitted model
        if META_MODEL_PATH.exists():
            try:
                with open(META_MODEL_PATH, "rb") as f:
                    payload = pickle.load(f)
                self._model = payload["model"]
                self._feature_names = payload["feature_names"]
                self._is_fitted = payload["is_fitted"]
                self._model_auc = payload.get("model_auc", 0.0)
                self._model_brier = payload.get("model_brier", 0.25)
                self._baseline_win_rate = payload.get("baseline_win_rate", 0.5)
                logger.info(
                    f"[Meta] Loaded fitted meta-learner | "
                    f"AUC={self._model_auc:.3f} | Brier={self._model_brier:.4f}"
                )
            except Exception as e:
                logger.warning(f"[Meta] Failed to load model: {e}")