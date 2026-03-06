"""
Regime detector — the most valuable single prediction in this system.

Markets alternate between regimes:
  - TRENDING (up/down): momentum strategies profit, reversion gets crushed
  - RANGING: reversion profits, trend following bleeds via whipsaws
  - VOLATILE: all strategies need tighter risk, or sit out

Knowing which regime you're in determines which strategy to weight up.

Architecture:
  Cold start: rule-based classification (ADX + MA alignment + volatility)
  After 50+ resolved trades: gradient boosting trained on features → regime labels
  Labels come from hindsight: was the next N bars trending or ranging?
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
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from core.models import Regime

MODEL_PATH = Path(__file__).parent.parent / "logs" / "regime_model.pkl"
DATA_PATH = Path(__file__).parent.parent / "logs" / "regime_data.json"
MIN_SAMPLES = 100
RETRAIN_EVERY = 50


class RegimeDetector:
    """
    Classifies market regime from technical features.

    Cold start: heuristic rules (ADX/BB/MA-based).
    After training: gradient boosting on labeled feature vectors.

    Labels are generated retroactively: after N bars, was the market
    trending (price moved > 2 ATR in one direction) or ranging?
    """

    def __init__(self, suffix: str = ""):
        self._model = None
        self._is_fitted = False
        self._feature_names: list[str] = []
        self._training_data: list[dict] = []
        self._samples_since_refit = 0
        self._accuracy: float = 0.0
        self._suffix = suffix
        self._model_path = Path(str(MODEL_PATH).replace(".pkl", f"{suffix}.pkl")) if suffix else MODEL_PATH
        self._data_path = Path(str(DATA_PATH).replace(".json", f"{suffix}.json")) if suffix else DATA_PATH
        self._load()

    def predict(self, features: dict[str, float]) -> tuple[Regime, float]:
        """
        Predict current regime.
        Returns: (regime, confidence)
        """
        if self._is_fitted and self._model is not None:
            try:
                X = np.array([[features.get(k, 0.0) for k in self._feature_names]])
                proba = self._model.predict_proba(X)[0]
                classes = self._model.classes_
                best_idx = int(np.argmax(proba))
                regime = Regime(classes[best_idx])
                confidence = float(proba[best_idx])
                return regime, round(confidence, 4)
            except Exception as e:
                logger.debug(f"[Regime] Model prediction failed: {e}")

        return self._heuristic(features)

    def add_labeled_sample(self, features: dict[str, float], regime: Regime):
        """
        Add a labeled sample (features at time T, actual regime over next N bars).
        Labels are assigned retroactively by examining price action after the fact.
        """
        self._training_data.append({
            "features": features,
            "label": regime.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._samples_since_refit += 1

        if (
            len(self._training_data) >= MIN_SAMPLES
            and self._samples_since_refit >= RETRAIN_EVERY
        ):
            self.refit()

        self._save_data()

    def label_from_hindsight(
        self,
        features_at_entry: dict[str, float],
        subsequent_closes: list[float],
        atr_at_entry: float,
    ):
        """
        Determine regime label from what actually happened after the features
        were observed. This is the key insight: we don't predict regime from
        current data alone — we LABEL it from future data, then train the model
        to predict those labels from current features.
        """
        if len(subsequent_closes) < 10 or atr_at_entry <= 0:
            return

        prices = np.array(subsequent_closes)
        start = prices[0]
        end = prices[-1]
        high = np.max(prices)
        low = np.min(prices)
        net_move = abs(end - start)
        total_range = high - low
        returns = np.diff(prices) / prices[:-1]

        # Was the move directional (trending) or choppy (ranging)?
        directional_ratio = net_move / total_range if total_range > 0 else 0
        move_in_atr = net_move / atr_at_entry
        range_in_atr = total_range / atr_at_entry

        # High volatility: total range > 4 ATR
        if range_in_atr > 4.0:
            regime = Regime.VOLATILE
        # Strong directional move: net > 2 ATR AND directional ratio > 0.5
        elif move_in_atr > 2.0 and directional_ratio > 0.5:
            regime = Regime.TRENDING_UP if end > start else Regime.TRENDING_DOWN
        # Moderate directional: net > 1 ATR AND mostly one direction
        elif move_in_atr > 1.0 and directional_ratio > 0.4:
            regime = Regime.TRENDING_UP if end > start else Regime.TRENDING_DOWN
        else:
            regime = Regime.RANGING

        self.add_labeled_sample(features_at_entry, regime)

    def refit(self) -> bool:
        """Retrain the regime classifier on accumulated labeled data."""
        n = len(self._training_data)
        if n < MIN_SAMPLES:
            return False
        if not (SKLEARN_AVAILABLE or LGBM_AVAILABLE):
            return False

        X = np.array([
            [s["features"].get(k, 0.0) for k in self._get_feature_keys()]
            for s in self._training_data
        ])
        y = np.array([s["label"] for s in self._training_data])
        self._feature_names = self._get_feature_keys()

        # At least 2 classes needed
        unique = np.unique(y)
        if len(unique) < 2:
            return False

        if LGBM_AVAILABLE and n >= 300:
            model = lgb.LGBMClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                min_child_samples=max(5, n // 50),
                random_state=42, verbose=-1,
            )
        elif SKLEARN_AVAILABLE:
            model = GradientBoostingClassifier(
                n_estimators=80, max_depth=4,
                min_samples_leaf=max(3, n // 50),
                random_state=42,
            )
        else:
            return False

        # Cross-validate
        cv_folds = min(5, n // 30)
        if cv_folds >= 2:
            try:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
                self._accuracy = float(scores.mean())
                logger.info(
                    f"[Regime] CV accuracy: {self._accuracy:.1%} ± {scores.std():.1%} "
                    f"({n} samples, {len(unique)} classes)"
                )
            except Exception as e:
                logger.debug(f"[Regime] CV failed: {e}")

        model.fit(X, y)
        self._model = model
        self._is_fitted = True
        self._samples_since_refit = 0
        self._save_model()
        return True

    def get_status(self) -> dict:
        return {
            "is_fitted": self._is_fitted,
            "training_samples": len(self._training_data),
            "accuracy": round(self._accuracy, 4),
            "needs": max(0, MIN_SAMPLES - len(self._training_data)),
        }

    # ── Heuristic fallback ───────────────────────────────────────────────────

    def _heuristic(self, features: dict[str, float]) -> tuple[Regime, float]:
        adx = features.get("adx", 20)
        ma_align = features.get("ma_alignment", 0)
        atr_ratio = features.get("atr_ratio", 1.0)
        bb_width = features.get("bb_width", 0.02)

        if atr_ratio > 1.8 or bb_width > 0.06:
            return Regime.VOLATILE, 0.6

        if adx > 30 and abs(ma_align) > 0.6:
            r = Regime.TRENDING_UP if ma_align > 0 else Regime.TRENDING_DOWN
            return r, 0.7

        if adx > 20 and abs(ma_align) > 0.3:
            r = Regime.TRENDING_UP if ma_align > 0 else Regime.TRENDING_DOWN
            return r, 0.5

        return Regime.RANGING, 0.5

    # ── Persistence ──────────────────────────────────────────────────────────

    def _get_feature_keys(self) -> list[str]:
        if self._training_data:
            return sorted(self._training_data[0]["features"].keys())
        return []

    def _save_data(self):
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        self._data_path.write_text(json.dumps(self._training_data[-5000:], indent=2))

    def _save_model(self):
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "feature_names": self._feature_names,
                "is_fitted": self._is_fitted,
                "accuracy": self._accuracy,
            }, f)

    def _load(self):
        if self._data_path.exists():
            try:
                self._training_data = json.loads(self._data_path.read_text())
                logger.info(f"[Regime] Loaded {len(self._training_data)} labeled samples")
            except Exception:
                pass
        if self._model_path.exists():
            try:
                with open(self._model_path, "rb") as f:
                    p = pickle.load(f)
                self._model = p["model"]
                self._feature_names = p["feature_names"]
                self._is_fitted = p["is_fitted"]
                self._accuracy = p.get("accuracy", 0.0)
                logger.info(f"[Regime] Loaded model (accuracy={self._accuracy:.1%})")
            except Exception:
                pass
