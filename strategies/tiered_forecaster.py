"""
tiered_forecaster.py — 90% cost reduction via three-tier LLM pipeline.

Tier 1: Ollama (FREE) — fast screen on ALL markets
  → Cheap 8B model, compressed prompt, just needs a rough estimate
  → If |ollama_estimate - market_price| < TIER1_THRESHOLD: skip (no edge)
  → Cost: $0

Tier 2: Claude Haiku / GPT-4o-mini (CHEAP ~$0.15/1M) — validate shortlist
  → More careful analysis, used as quality gate before expensive models
  → If combined estimate still shows edge AND confidence > threshold: escalate
  → Cost: ~$0.02/market evaluated

Tier 3: Claude Sonnet + GPT-4o (EXPENSIVE) — only on high-confidence shortlist
  → Full superforecaster prompt, used for actual trade signals
  → Typically only 5-10% of all markets reach this tier
  → Cost: ~$0.30/market evaluated

Typical flow:
  300 markets → 30 pass Tier 1 → 10 pass Tier 2 → Tier 3 ensemble on 10 markets

Plus:
  - LLM result cache (skip re-evaluation of unchanged markets)
  - Token budget compression (trim news to stay in limits)
  - Per-provider cost tracking
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import statistics
import time
from typing import Optional

import httpx
from loguru import logger

from core.config import config
from core.models import EnrichedMarket, EnsembleEstimate, LLMEstimate
from utils.ml_calibration import MLCalibrator


# ─── Thresholds for tier escalation ──────────────────────────────────────────
# Tier 1 → Tier 2: Ollama sees potential edge above this
TIER1_MIN_EDGE = 0.06          # 6% raw edge from Ollama to escalate

# Tier 2 → Tier 3: cheap model confirms edge, confidence high enough
TIER2_MIN_EDGE = 0.05          # 5% edge after cheap model confirms
TIER2_MIN_CONFIDENCE = 0.55    # Cheap model must be reasonably confident

# Tier 3: full ensemble threshold (set by bot config: MIN_EDGE, MIN_LLM_CONFIDENCE)


# ─── Tier 1: Minimal prompt (fast, cheap) ────────────────────────────────────
TIER1_SYSTEM = """You are a prediction market analyst. Estimate the probability a market
resolves YES. Respond ONLY with valid JSON: {"probability": 0.XX, "confidence": 0.XX}"""

TIER1_USER = """Market: {question}
Resolution: {resolution}
Current price: {market_price:.2%}
Recent context: {news_brief}
Provide your probability estimate as JSON."""


# ─── Tier 2: Medium prompt (balanced) ────────────────────────────────────────
TIER2_SYSTEM = """You are a careful probabilistic forecaster. Estimate the probability
a prediction market resolves YES. Consider base rates and recent evidence.
Respond ONLY with JSON: {"probability": 0.XX, "confidence": 0.XX, "reasoning": "brief"}"""

TIER2_USER = """Market: {question}
Resolution criteria: {resolution}
Current market price: {market_price:.2%}
Recent news:
{news_summary}
Provide a calibrated probability estimate."""


# ─── Tier 3: Full superforecaster prompt (matches original forecaster.py) ─────
TIER3_SYSTEM = """You are an expert superforecaster trained in the style of Philip Tetlock.
Your goal is to maximize accuracy by minimizing Brier scores.

Process:
1. Identify reference classes and establish base rates (outside view)
2. List strongest evidence FOR and AGAINST higher probability (inside view)
3. Adjust base rate for distinctive features of this situation
4. Consider resolution criteria carefully — markets often resolve differently than expected
5. Avoid overconfidence — extreme probabilities require extraordinary evidence

Respond ONLY with valid JSON:
{
  "probability": 0.XX,
  "confidence": 0.XX,
  "base_rate": 0.XX,
  "reasoning": "2-4 sentences",
  "key_uncertainty": "biggest factor that could make you wrong"
}"""

TIER3_USER = """MARKET: {question}

RESOLUTION CRITERIA: {resolution}

CURRENT MARKET PRICE (crowd estimate): {market_price:.2%}

RECENT RELEVANT NEWS:
{news_summary}

TIME TO RESOLUTION: {time_to_resolution}

Provide your calibrated probability estimate."""


class TieredForecaster:
    """
    Three-tier LLM pipeline. Processes every market through Ollama (free),
    escalates promising ones to cheap cloud LLMs, and only uses expensive
    frontier models on the strongest candidates.
    """

    def __init__(self):
        # Track per-provider weights (updated by calibration feedback)
        self._provider_weights = {
            "openai_mini": 0.8,
            "openai": 1.0,
            "anthropic_haiku": 0.8,
            "anthropic": 1.0,
            "ollama": 0.7,
        }

        # ML calibration (Platt scaling + per-category bias correction)
        # Falls back to fixed shrinkage until 30+ resolved trades accumulate
        self._ml_calibrator = MLCalibrator(
            fallback_shrinkage=config.bot.llm_shrinkage
        )
        self._cache: dict[str, tuple[float, EnsembleEstimate]] = {}
        self._cache_ttl_seconds = 3600  # 1 hour — re-evaluate if market moves

        # Cost tracking (USD)
        self._session_costs = {
            "ollama": 0.0,
            "openai_mini": 0.0,
            "openai": 0.0,
            "anthropic_haiku": 0.0,
            "anthropic": 0.0,
        }
        self._session_calls = dict.fromkeys(self._session_costs, 0)

    # ─── Public interface ─────────────────────────────────────────────────────

    async def estimate(self, enriched: EnrichedMarket) -> Optional[EnsembleEstimate]:
        """
        Run tiered evaluation. Returns None if no edge found at any tier.
        Returns EnsembleEstimate if the market passes all three tiers.
        """
        market = enriched.market

        # Check cache first
        cache_key = self._cache_key(enriched)
        cached = self._get_cached(cache_key, market.yes_price)
        if cached:
            logger.debug(f"[Tier] Cache hit: {market.question[:50]}")
            return cached

        # ── Tier 1: Ollama screen ─────────────────────────────────────────────
        if not config.llm.ollama_enabled:
            # Skip tier 1 if Ollama unavailable — go straight to Tier 2/3
            logger.debug("[Tier] Ollama unavailable — skipping Tier 1 screen")
            tier1_result = None
        else:
            tier1_result = await self._tier1_ollama(enriched)

            if tier1_result is not None:
                tier1_edge = abs(tier1_result.probability - market.yes_price)
                if tier1_edge < TIER1_MIN_EDGE:
                    logger.debug(
                        f"[Tier1] ✗ No edge ({tier1_edge:.1%}): {market.question[:55]}"
                    )
                    return None
                logger.debug(
                    f"[Tier1] ✓ Edge {tier1_edge:.1%} → escalating: {market.question[:55]}"
                )

        # ── Tier 2: Cheap cloud LLM validation ───────────────────────────────
        tier2_result = await self._tier2_cheap(enriched)

        if tier2_result is not None:
            tier2_edge = abs(tier2_result.probability - market.yes_price)
            if tier2_edge < TIER2_MIN_EDGE or tier2_result.confidence < TIER2_MIN_CONFIDENCE:
                logger.debug(
                    f"[Tier2] ✗ Failed (edge={tier2_edge:.1%} conf={tier2_result.confidence:.0%}): "
                    f"{market.question[:50]}"
                )
                return None
            logger.debug(
                f"[Tier2] ✓ Confirmed → escalating to Tier 3: {market.question[:50]}"
            )

        # ── Tier 3: Full ensemble on shortlisted market ───────────────────────
        logger.info(f"[Tier3] 🎯 Full evaluation: {market.question[:60]}")
        ensemble = await self._tier3_full(enriched, tier1_result, tier2_result)

        if ensemble:
            self._cache_result(cache_key, ensemble)

        return ensemble

    def get_ml_calibration_report(self) -> dict:
        """Return ML calibration model status and metrics."""
        return self._ml_calibrator.get_calibration_report()

    def record_resolved_outcome(
        self, llm_prob: float, actual_outcome: float, category: str = "unknown"
    ):
        """Feed resolved trade back to the ML calibrator for retraining."""
        self._ml_calibrator.add_outcome(llm_prob, actual_outcome, category)
        self._ml_calibrator.maybe_refit()

    def get_cost_summary(self) -> dict:
        """Return cost tracking for the current session."""
        total = sum(self._session_costs.values())
        return {
            "total_usd": round(total, 4),
            "by_provider": {k: round(v, 4) for k, v in self._session_costs.items()},
            "calls_by_provider": dict(self._session_calls),
            "cache_size": len(self._cache),
            "tip": f"Annualised cost at this rate: ~${total * 365:.0f}/year" if total > 0 else "No costs yet",
        }

    # ─── Tier implementations ─────────────────────────────────────────────────

    async def _tier1_ollama(self, enriched: EnrichedMarket) -> Optional[LLMEstimate]:
        """Tier 1: Fast Ollama screen. Compressed prompt, minimal tokens."""
        market = enriched.market
        # Only use first 200 chars of news for Tier 1 — keep it fast
        news_brief = enriched.news_summary[:200] if enriched.news_summary else "No recent news."

        user_prompt = TIER1_USER.format(
            question=market.question,
            resolution=(market.resolution_criteria or "Standard resolution")[:150],
            market_price=market.yes_price,
            news_brief=news_brief,
        )

        url = f"{config.llm.ollama_base_url}/api/chat"
        payload = {
            "model": config.llm.ollama_model,
            "messages": [
                {"role": "system", "content": TIER1_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 100},
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            raw = data.get("message", {}).get("content", "")
            est = self._parse_estimate("ollama", config.llm.ollama_model, raw)
            self._track_cost("ollama", input_tokens=300, output_tokens=80)
            return est

        except Exception as e:
            logger.debug(f"[Tier1] Ollama failed: {e}")
            return None

    async def _tier2_cheap(self, enriched: EnrichedMarket) -> Optional[LLMEstimate]:
        """Tier 2: Cheap cloud LLM. Use mini/haiku models."""
        market = enriched.market
        news_summary = self._compress_news(enriched.news_summary, max_chars=500)

        user_prompt = TIER2_USER.format(
            question=market.question,
            resolution=(market.resolution_criteria or "Not specified")[:200],
            market_price=market.yes_price,
            news_summary=news_summary,
        )

        # Try cheapest available provider first
        if config.llm.anthropic_enabled:
            result = await self._call_anthropic(
                user_prompt, TIER2_SYSTEM,
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                provider_key="anthropic_haiku",
            )
            if result:
                self._track_cost("anthropic_haiku", input_tokens=600, output_tokens=120)
                return result

        if config.llm.openai_enabled:
            result = await self._call_openai(
                user_prompt, TIER2_SYSTEM,
                model="gpt-4o-mini",
                max_tokens=150,
                provider_key="openai_mini",
            )
            if result:
                self._track_cost("openai_mini", input_tokens=600, output_tokens=120)
                return result

        return None

    async def _tier3_full(
        self,
        enriched: EnrichedMarket,
        tier1: Optional[LLMEstimate],
        tier2: Optional[LLMEstimate],
    ) -> Optional[EnsembleEstimate]:
        """Tier 3: Full ensemble with frontier models."""
        market = enriched.market
        news_summary = self._compress_news(enriched.news_summary, max_chars=1500)

        time_to_res = self._format_time_to_resolution(market)
        user_prompt = TIER3_USER.format(
            question=market.question,
            resolution=(market.resolution_criteria or "Not specified")[:400],
            market_price=market.yes_price,
            news_summary=news_summary,
            time_to_resolution=time_to_res,
        )

        # Launch frontier models concurrently
        tasks = []
        if config.llm.anthropic_enabled:
            tasks.append(self._call_anthropic(
                user_prompt, TIER3_SYSTEM,
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                provider_key="anthropic",
            ))
        if config.llm.openai_enabled:
            tasks.append(self._call_openai(
                user_prompt, TIER3_SYSTEM,
                model="gpt-4o",
                max_tokens=400,
                provider_key="openai",
            ))

        if not tasks:
            # Fallback: use Tier 2 result as final estimate if no frontier models
            if tier2:
                return self._single_to_ensemble([tier2], market.yes_price)
            return None

        results = await asyncio.gather(*tasks, return_exceptions=True)
        tier3_estimates = [r for r in results if isinstance(r, LLMEstimate)]

        # Track costs for frontier models
        if config.llm.anthropic_enabled:
            self._track_cost("anthropic", input_tokens=1400, output_tokens=350)
        if config.llm.openai_enabled:
            self._track_cost("openai", input_tokens=1400, output_tokens=350)

        # Combine: all tiers contribute, weighted by tier
        all_estimates = []
        if tier1:
            # Tier 1 gets lower weight (cruder model)
            tier1.confidence *= 0.6
            all_estimates.append(tier1)
        if tier2:
            # Tier 2 medium weight
            tier2.confidence *= 0.8
            all_estimates.append(tier2)
        all_estimates.extend(tier3_estimates)

        if not all_estimates:
            return None

        return self._build_ensemble(all_estimates, market.yes_price)

    # ─── LLM API callers ──────────────────────────────────────────────────────

    async def _call_anthropic(
        self, user_prompt: str, system: str,
        model: str, max_tokens: int, provider_key: str
    ) -> Optional[LLMEstimate]:
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=config.llm.anthropic_api_key)
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system + "\n\nRespond ONLY with valid JSON, no other text.",
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.1,
            )
            raw = response.content[0].text
            self._session_calls[provider_key] = self._session_calls.get(provider_key, 0) + 1
            return self._parse_estimate(provider_key, model, raw)
        except Exception as e:
            logger.warning(f"[Tier] Anthropic ({model}) failed: {e}")
            return None

    async def _call_openai(
        self, user_prompt: str, system: str,
        model: str, max_tokens: int, provider_key: str
    ) -> Optional[LLMEstimate]:
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=config.llm.openai_api_key)
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            self._session_calls[provider_key] = self._session_calls.get(provider_key, 0) + 1
            return self._parse_estimate(provider_key, model, raw)
        except Exception as e:
            logger.warning(f"[Tier] OpenAI ({model}) failed: {e}")
            return None

    # ─── Parsing + ensemble ───────────────────────────────────────────────────

    def _parse_estimate(self, provider: str, model: str, raw: str) -> Optional[LLMEstimate]:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        json_match = re.search(r"\{.*\}", clean, re.DOTALL)
        if json_match:
            clean = json_match.group()
        try:
            data = json.loads(clean)
            prob = max(0.01, min(0.99, float(data.get("probability", 0.5))))
            conf = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            return LLMEstimate(
                provider=provider,
                model=model,
                probability=prob,
                confidence=conf,
                base_rate=float(data["base_rate"]) if data.get("base_rate") else None,
                reasoning=data.get("reasoning", ""),
                raw_response=raw,
            )
        except Exception as e:
            logger.debug(f"[Tier] Parse failed ({provider}): {e}")
            return None

    def _build_ensemble(
        self, estimates: list[LLMEstimate], market_price: float
    ) -> EnsembleEstimate:
        """Weighted ensemble with shrinkage and crowd blend."""
        total_w, weighted_sum = 0.0, 0.0
        for est in estimates:
            w = self._provider_weights.get(est.provider, 1.0) * est.confidence
            weighted_sum += est.probability * w
            total_w += w

        ensemble_prob = weighted_sum / total_w if total_w > 0 else 0.5
        avg_conf = statistics.mean(e.confidence for e in estimates)

        # Shrinkage: ML-fitted Platt scaling if available, else fixed shrinkage
        category = "unknown"  # Will be passed in from context in future
        calibrated = self._ml_calibrator.calibrate(
            raw_prob=ensemble_prob,
            category=category,
        )

        # Crowd blend (NeurIPS 2024 finding)
        final = config.bot.llm_weight * calibrated + config.bot.crowd_weight * market_price

        consensus = statistics.stdev([e.probability for e in estimates]) if len(estimates) > 1 else 0.0
        reasoning = " | ".join(
            f"[{e.provider.upper()}] {e.reasoning}" for e in estimates if e.reasoning
        )

        return EnsembleEstimate(
            raw_estimates=estimates,
            ensemble_probability=round(ensemble_prob, 4),
            calibrated_probability=round(calibrated, 4),
            final_probability=round(final, 4),
            confidence=round(avg_conf, 4),
            consensus=round(consensus, 4),
            reasoning_summary=reasoning[:1000],
        )

    def _single_to_ensemble(
        self, estimates: list[LLMEstimate], market_price: float
    ) -> EnsembleEstimate:
        """Wrap a single estimate as an ensemble (fallback)."""
        return self._build_ensemble(estimates, market_price)

    # ─── Cost tracking ────────────────────────────────────────────────────────

    # Pricing per 1M tokens (approximate, March 2026)
    _TOKEN_COSTS = {
        "ollama":           (0.000, 0.000),
        "openai_mini":      (0.150, 0.600),   # gpt-4o-mini
        "openai":           (2.500, 10.000),  # gpt-4o
        "anthropic_haiku":  (0.080, 0.400),   # claude-haiku-4-5
        "anthropic":        (3.000, 15.000),  # claude-sonnet-4
    }

    def _track_cost(self, provider: str, input_tokens: int, output_tokens: int):
        in_rate, out_rate = self._TOKEN_COSTS.get(provider, (0, 0))
        cost = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
        self._session_costs[provider] = self._session_costs.get(provider, 0.0) + cost
        self._session_calls[provider] = self._session_calls.get(provider, 0) + 1

    # ─── Cache ────────────────────────────────────────────────────────────────

    def _cache_key(self, enriched: EnrichedMarket) -> str:
        """Cache key: market ID + current price (invalidates on price change)."""
        key = f"{enriched.market.question_id}:{enriched.market.yes_price:.4f}"
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cached(self, key: str, current_price: float) -> Optional[EnsembleEstimate]:
        entry = self._cache.get(key)
        if not entry:
            return None
        timestamp, estimate = entry
        if time.time() - timestamp > self._cache_ttl_seconds:
            del self._cache[key]
            return None
        return estimate

    def _cache_result(self, key: str, estimate: EnsembleEstimate):
        self._cache[key] = (time.time(), estimate)
        # Evict oldest entries if cache grows large
        if len(self._cache) > 2000:
            oldest = sorted(self._cache.items(), key=lambda x: x[1][0])[:500]
            for k, _ in oldest:
                del self._cache[k]

    # ─── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _compress_news(summary: str, max_chars: int) -> str:
        """Trim news summary to token budget while keeping most recent items."""
        if not summary or len(summary) <= max_chars:
            return summary or "No recent news."
        # Keep first line (market header) + as many news items as fit
        lines = summary.split("\n")
        result = []
        chars = 0
        for line in lines:
            if chars + len(line) + 1 > max_chars:
                break
            result.append(line)
            chars += len(line) + 1
        return "\n".join(result)

    @staticmethod
    def _format_time_to_resolution(market) -> str:
        if not market.close_time:
            return "unknown"
        from datetime import datetime, timezone
        delta = market.close_time - datetime.now(timezone.utc)
        days = delta.days
        if days < 0:
            return "resolving soon"
        if days == 0:
            return "resolves today"
        if days == 1:
            return "resolves tomorrow"
        return f"resolves in {days} days"

    def update_provider_weight(self, provider: str, brier_score: float):
        """Adaptive reweighting based on resolved trade outcomes."""
        current = self._provider_weights.get(provider, 1.0)
        new_w = max(0.2, min(1.5, 1.0 - brier_score))
        self._provider_weights[provider] = 0.8 * current + 0.2 * new_w
        logger.info(f"[Tier] Weight update {provider}: {current:.3f} → {self._provider_weights[provider]:.3f}")