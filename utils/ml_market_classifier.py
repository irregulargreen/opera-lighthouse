"""
ml_market_classifier.py — gradient boosting classifier to screen markets.

Replaces / augments the Ollama Tier 1 screen with a trained ML model that
predicts whether a market is worth LLM evaluation.

Specifically predicts: P(|final_price - market_price| > MIN_EDGE at resolution)
i.e. "is this market likely to be mispriced right now?"

Features extracted from market metadata alone (no LLM needed):
  - Liquidity (raw + log)
  - Volume 24h (raw + log)
  - Volume total
  - Spread
  - Time to close (days)
  - Category (one-hot encoded)
  - Current price (and distance from 0.5)
  - Price × time interaction
  - Is new market (bool)
  - Volume velocity (24h / total ratio)

Why gradient boosting (LightGBM/XGBoost)?
  - Excellent on small tabular datasets (300-2000 samples)
  - Handles missing values natively
  - No feature scaling required
  - Fast inference: <1ms per prediction
  - Interpretable feature importances

Training data source:
  Option A: Bootstrap from Polymarket historical data API
            (fetch resolved markets + their final prices vs open prices)
  Option B: Accumulate from the bot's own calibration.db over time

The classifier runs BEFORE Ollama — completely free, <1ms per market.
"""
from __future__ import annotations

import json
import pickle
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
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression as SklearnLR
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from core.models import Market


MODEL_PATH = Path(__file__).parent.parent / "logs" / "market_classifier.pkl"
MIN_TRAINING_SAMPLES = 200  # Before this: fall back to heuristic rules


# Market categories (for one-hot encoding)
KNOWN_CATEGORIES = [
    "politics", "crypto", "sports", "finance", "science",
    "entertainment", "weather", "other"
]


class MarketClassifier:
    """
    Gradient boosting classifier that screens markets for LLM evaluation.

    Prediction: P(market is mispriced by > MIN_EDGE) based on market features alone.
    Use as a fast, free pre-filter before any LLM calls.

    >>> clf = MarketClassifier()
    >>> score = clf.predict_mispricing_score(market)
    >>> if score > 0.3:  # 30% probability of mispricing
    ...     # Escalate to LLM
    """

    def __init__(self, min_edge: float = 0.05):
        self.min_edge = min_edge
        self._model = None
        self._label_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        self._feature_names: list[str] = []
        self._is_fitted = False
        self._training_data: list[dict] = []

        self._load_from_disk()

    # ─── Public interface ─────────────────────────────────────────────────────

    def predict_mispricing_score(self, market: Market) -> float:
        """
        Returns P(mispricing > MIN_EDGE) for this market.
        Range: 0.0 (definitely efficiently priced) to 1.0 (likely mispriced).

        If model not fitted yet: uses heuristic scoring rules.
        """
        features = self._extract_features(market)

        if self._is_fitted and self._model is not None:
            X = np.array([list(features.values())])
            try:
                prob = float(self._model.predict_proba(X)[0][1])
                return round(prob, 4)
            except Exception as e:
                logger.debug(f"[MLScreen] Model predict failed: {e}")

        # Fallback: heuristic rules when model not yet fitted
        return self._heuristic_score(market)

    def should_evaluate(self, market: Market, threshold: float = 0.25) -> bool:
        """
        Returns True if the market is worth sending to LLM evaluation.
        Threshold: 0.25 = pass if >25% chance of mispricing.
        """
        return self.predict_mispricing_score(market) >= threshold

    def add_training_sample(
        self,
        market: Market,
        was_mispriced: bool,
        final_price: Optional[float] = None,
        our_probability: Optional[float] = None,
    ):
        """
        Record whether a market was actually mispriced (for training).

        was_mispriced:   True if |final_resolution_price - open_price| > min_edge
        final_price:     The price at resolution (1.0 or 0.0)
        our_probability: The LLM's probability estimate (if available)
        """
        features = self._extract_features(market)
        sample = {
            "features": features,
            "label": int(was_mispriced),
            "final_price": final_price,
            "our_probability": our_probability,
            "market_id": market.question_id,
        }
        self._training_data.append(sample)

    def fit(self, min_samples: int = MIN_TRAINING_SAMPLES) -> bool:
        """
        Fit the classifier on accumulated training data.
        Returns True if fitting succeeded.
        """
        n = len(self._training_data)
        if n < min_samples:
            logger.info(f"[MLScreen] Not enough samples ({n}/{min_samples}) — using heuristics")
            return False

        if not SKLEARN_AVAILABLE:
            logger.warning("[MLScreen] sklearn not available — cannot fit model")
            return False

        X = np.array([list(s["features"].values()) for s in self._training_data])
        y = np.array([s["label"] for s in self._training_data])
        self._feature_names = list(self._training_data[0]["features"].keys())

        # Class balance check
        pos_rate = y.mean()
        logger.info(f"[MLScreen] Training on {n} samples | mispricing rate: {pos_rate:.1%}")

        # Choose model based on data size and available libraries
        if LGBM_AVAILABLE and n >= 500:
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",  # Handle class imbalance
                random_state=42,
                verbose=-1,
            )
        elif SKLEARN_AVAILABLE and n >= 200:
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
        else:
            # Minimal data: simple logistic regression
            model = SklearnLR(C=1.0, class_weight="balanced", random_state=42)

        # Cross-validation
        try:
            scores = cross_val_score(model, X, y, cv=min(5, n // 40), scoring="roc_auc")
            logger.info(f"[MLScreen] Cross-val AUC: {scores.mean():.3f} ± {scores.std():.3f}")
        except Exception as e:
            logger.debug(f"[MLScreen] CV failed: {e}")

        # Fit final model
        model.fit(X, y)
        self._model = model
        self._is_fitted = True

        # Log feature importances
        self._log_feature_importances()

        self._save_to_disk()
        return True

    def get_feature_importances(self) -> dict[str, float]:
        """Return feature importance scores from the fitted model."""
        if not self._is_fitted or self._model is None:
            return {}
        try:
            if hasattr(self._model, "feature_importances_"):
                importances = self._model.feature_importances_
                return dict(zip(self._feature_names, [round(float(i), 4) for i in importances]))
        except Exception:
            pass
        return {}

    # ─── Feature engineering ──────────────────────────────────────────────────

    def _extract_features(self, market: Market) -> dict[str, float]:
        """
        Extract numerical features from a Market object.
        All features must be computable from market metadata alone.
        """
        from datetime import datetime, timezone

        # Time to close
        if market.close_time:
            days_to_close = max(0, (market.close_time - datetime.now(timezone.utc)).days)
        else:
            days_to_close = 30.0  # Unknown → assume medium term

        # Volume velocity: how much of total volume was recent?
        if market.volume_total > 0:
            volume_velocity = market.volume_24h / market.volume_total
        else:
            volume_velocity = 0.0

        # Price features
        price = market.yes_price
        price_distance_from_half = abs(price - 0.5)
        # Markets near 50/50 are most interesting (more uncertainty → more edge opportunities)
        near_midpoint = float(price_distance_from_half < 0.20)

        # Liquidity features (log to handle wide range)
        liq = max(1.0, market.liquidity)
        vol_24h = max(1.0, market.volume_24h)
        vol_total = max(1.0, market.volume_total)

        # Category one-hot (known categories)
        cat = (market.category or "other").lower()
        # Normalise common variations
        cat = "politics" if any(x in cat for x in ["polit", "elect", "gov"]) else cat
        cat = "crypto" if any(x in cat for x in ["crypto", "btc", "eth", "coin"]) else cat
        cat = "sports" if any(x in cat for x in ["sport", "nba", "nfl", "soccer"]) else cat
        cat_features = {f"cat_{c}": float(cat == c) for c in KNOWN_CATEGORIES}

        features = {
            # Liquidity (log-transformed — ranges from $10 to $10M)
            "log_liquidity": float(np.log1p(liq)),
            "liquidity_raw": float(min(liq, 1_000_000)),

            # Volume
            "log_volume_24h": float(np.log1p(vol_24h)),
            "log_volume_total": float(np.log1p(vol_total)),
            "volume_velocity": float(volume_velocity),

            # Price features
            "yes_price": float(price),
            "price_distance_from_half": float(price_distance_from_half),
            "near_midpoint": near_midpoint,

            # Spread (wider spread = less efficient = more opportunity)
            "spread": float(min(market.spread, 0.5)),

            # Time features
            "days_to_close": float(min(days_to_close, 365)),
            "log_days_to_close": float(np.log1p(days_to_close)),
            "is_closing_soon": float(days_to_close < 7),
            "is_new_market": float(market.is_new),

            # Interaction: price × time (long-dated near-50/50 = most uncertain)
            "price_time_interaction": float(price_distance_from_half * np.log1p(days_to_close)),

            **cat_features,
        }

        return features

    # ─── Heuristics (pre-model fallback) ─────────────────────────────────────

    def _heuristic_score(self, market: Market) -> float:
        """
        Rule-based score when model not fitted.
        Encodes domain knowledge about which markets tend to be mispriced.
        """
        from datetime import datetime, timezone

        score = 0.3  # Base rate: ~30% of markets have some mispricing

        # New markets: highest edge window
        if market.is_new:
            score += 0.25

        # Near midpoint: more uncertainty → more room for LLM to find edge
        if 0.30 <= market.yes_price <= 0.70:
            score += 0.10

        # Sweet spot liquidity ($1k - $100k): enough to trade, not efficiently priced
        if 1_000 <= market.liquidity <= 100_000:
            score += 0.10
        elif market.liquidity > 500_000:
            score -= 0.15  # Over-liquid = efficiently priced by pros

        # Wide spread = less competition
        if market.spread > 0.05:
            score += 0.10

        # Low volume in last 24h relative to total = stale pricing
        if market.volume_total > 0 and market.volume_24h / market.volume_total < 0.05:
            score += 0.08  # Market hasn't been actively traded recently

        # Long time horizon = more room for information to arrive
        if market.close_time:
            days = (market.close_time - datetime.now(timezone.utc)).days
            if 7 <= days <= 90:
                score += 0.05

        return round(min(0.95, max(0.0, score)), 4)

    # ─── Persistence ──────────────────────────────────────────────────────────

    def _save_to_disk(self):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted,
            "training_data": self._training_data[-5000:],
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"[MLScreen] Saved model ({len(self._training_data)} samples)")

    def _load_from_disk(self):
        if not MODEL_PATH.exists():
            return
        try:
            with open(MODEL_PATH, "rb") as f:
                payload = pickle.load(f)
            self._model = payload.get("model")
            self._feature_names = payload.get("feature_names", [])
            self._is_fitted = payload.get("is_fitted", False)
            self._training_data = payload.get("training_data", [])
            logger.info(
                f"[MLScreen] Loaded classifier | fitted={self._is_fitted} | "
                f"training_samples={len(self._training_data)}"
            )
        except Exception as e:
            logger.warning(f"[MLScreen] Failed to load classifier: {e}")

    def _log_feature_importances(self):
        importances = self.get_feature_importances()
        if importances:
            top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"[MLScreen] Top features: {top5}")


# ─── Bootstrap from Polymarket historical data ───────────────────────────────

async def bootstrap_from_historical(
    classifier: MarketClassifier,
    min_edge: float = 0.05,
    n_markets: int = 1000,
):
    """
    Seed the classifier with historical Polymarket data.

    Fetches recently resolved markets from the Gamma API and labels them:
    - Was the opening price more than min_edge away from final resolution (0 or 1)?

    This bootstraps the ML model without needing to wait for your own trades to resolve.
    """
    import aiohttp
    from datetime import datetime, timedelta, timezone
    from data.scanner import MarketScanner

    logger.info(f"[MLScreen] Bootstrapping from {n_markets} historical markets...")

    async with aiohttp.ClientSession() as session:
        # Fetch recently resolved markets
        url = "https://gamma-api.polymarket.com/markets"
        params = {
            "closed": "true",
            "order": "endDate",
            "ascending": "false",
            "limit": min(n_markets, 100),
        }

        all_raw = []
        offset = 0
        while len(all_raw) < n_markets:
            params["offset"] = offset
            async with session.get(url, params=params) as resp:
                data = await resp.json()
            if not data:
                break
            all_raw.extend(data)
            offset += len(data)
            if len(data) < 100:
                break

        logger.info(f"[MLScreen] Fetched {len(all_raw)} resolved markets")

        for raw in all_raw[:n_markets]:
            try:
                # Parse market
                tokens = raw.get("tokens", [])
                if not tokens:
                    continue

                import json as _json
                prices = _json.loads(raw.get("outcomePrices", "[0.5,0.5]") or "[0.5,0.5]")
                open_price = float(prices[0]) if prices else 0.5

                # Determine resolution (YES = 1.0, NO = 0.0)
                # Resolved markets have outcomePrices close to 0 or 1
                resolved_price = open_price
                for token in tokens:
                    if token.get("outcome", "").upper() == "YES":
                        # Check if winner field is set
                        if token.get("winner"):
                            resolved_price = 1.0
                        elif raw.get("resolutionSource") == "no":
                            resolved_price = 0.0
                        break

                final_price = resolved_price
                was_mispriced = abs(final_price - open_price) > min_edge

                # Build a minimal Market object for feature extraction
                from core.models import Market as MarketModel
                from datetime import datetime, timezone
                market = MarketModel(
                    condition_id=raw.get("conditionId", ""),
                    question_id=raw.get("id", ""),
                    token_id_yes=tokens[0].get("token_id", "") if tokens else "",
                    token_id_no=tokens[1].get("token_id", "") if len(tokens) > 1 else "",
                    question=raw.get("question", ""),
                    category=raw.get("category"),
                    yes_price=open_price,
                    no_price=1 - open_price,
                    volume_24h=float(raw.get("volume24hr", 0) or 0),
                    volume_total=float(raw.get("volume", 0) or 0),
                    liquidity=float(raw.get("liquidity", 0) or 0),
                    spread=float(raw.get("spread", 0.02) or 0.02),
                )

                classifier.add_training_sample(
                    market=market,
                    was_mispriced=was_mispriced,
                    final_price=final_price,
                )

            except Exception as e:
                logger.debug(f"[MLScreen] Failed to parse market: {e}")

    logger.info(f"[MLScreen] Bootstrap complete: {len(classifier._training_data)} samples")

    # Fit if we have enough
    if len(classifier._training_data) >= MIN_TRAINING_SAMPLES:
        classifier.fit()
    else:
        logger.warning(
            f"[MLScreen] Only {len(classifier._training_data)} valid samples — "
            f"need {MIN_TRAINING_SAMPLES}. More historical data needed."
        )