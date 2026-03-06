"""
scanner.py — polls Gamma API for markets, enriches with CLOB order book data.

Two modes:
  1. NEW MARKET SWEEP: detects markets created in the last N minutes (high edge window)
  2. FULL SWEEP: scans all active markets filtered by volume/liquidity thresholds
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import config
from core.models import Market, OrderBook


GAMMA_API = config.polymarket.gamma_host
CLOB_API = config.polymarket.clob_host

# Markets created within this window are flagged as "new"
NEW_MARKET_WINDOW_MINUTES = 90


class MarketScanner:
    """
    Fetches and normalises markets from Polymarket's Gamma and CLOB APIs.
    All network calls are async to allow concurrent enrichment.
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"Accept": "application/json"},
        )
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    # ─── Public interface ─────────────────────────────────────────────────────

    async def get_new_markets(self) -> list[Market]:
        """
        Returns markets created in the last NEW_MARKET_WINDOW_MINUTES.
        These are the highest-edge opportunities — crowd hasn't had time to
        efficiently price them yet.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=NEW_MARKET_WINDOW_MINUTES)
        raw = await self._fetch_markets(
            params={
                "active": "true",
                "closed": "false",
                "order": "createdAt",
                "ascending": "false",
                "limit": 100,
            }
        )
        markets = [self._parse_market(m) for m in raw if m]
        new = []
        for m in markets:
            if m.created_at and m.created_at > cutoff:
                m.is_new = True
                new.append(m)
        logger.info(f"[Scanner] New markets in last {NEW_MARKET_WINDOW_MINUTES}min: {len(new)}")
        return new

    async def get_active_markets(
        self,
        min_volume_24h: float = 0.0,
        min_liquidity: float = 0.0,
        tag: Optional[str] = None,
        limit: int = 500,
    ) -> list[Market]:
        """
        Full sweep of active markets. Filters out highly liquid, efficiently
        priced markets (where edges are thin) and focuses on the middle tier.
        """
        params: dict = {
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
            "limit": limit,
        }
        if tag:
            params["tag"] = tag

        raw = await self._fetch_markets(params=params)
        markets = [self._parse_market(m) for m in raw if m]

        # Filter: must meet minimum thresholds
        filtered = [
            m for m in markets
            if m.volume_24h >= min_volume_24h
            and m.liquidity >= min_liquidity
        ]

        # Exclude top-tier hyper-liquid markets (they're efficiently priced by pros)
        # Focus on the "sweet spot": enough liquidity to trade, not enough to be efficient
        sweet_spot = [
            m for m in filtered
            if m.liquidity < 500_000  # Under $500k liquidity
        ]

        logger.info(
            f"[Scanner] Active markets: {len(markets)} total → "
            f"{len(filtered)} above thresholds → {len(sweet_spot)} in sweet spot"
        )
        return sweet_spot

    async def get_order_book(self, token_id: str) -> OrderBook:
        """Fetch and parse the CLOB order book for a single token."""
        url = f"{CLOB_API}/book"
        params = {"token_id": token_id}
        data = await self._get(url, params=params)
        return self._parse_order_book(token_id, data or {})

    async def enrich_with_order_books(self, markets: list[Market]) -> list[Market]:
        """
        Concurrently fetch order books for a list of markets and update
        their yes_price, no_price, spread, and liquidity fields.
        Uses semaphore to respect rate limits (300 req/10s on /books).
        """
        sem = asyncio.Semaphore(20)  # 20 concurrent book requests

        async def fetch_one(market: Market) -> Market:
            async with sem:
                try:
                    book = await self.get_order_book(market.token_id_yes)
                    market.yes_price = book.mid
                    market.no_price = round(1.0 - book.mid, 6)
                    market.spread = book.spread
                    # Update liquidity from live book depth
                    live_liq = book.depth_bid_usdc + book.depth_ask_usdc
                    if live_liq > 0:
                        market.liquidity = live_liq
                except Exception as e:
                    logger.warning(f"[Scanner] Order book failed for {market.question[:50]}: {e}")
                return market

        tasks = [fetch_one(m) for m in markets]
        return list(await asyncio.gather(*tasks))

    # ─── Gamma API fetching ───────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _fetch_markets(self, params: dict) -> list[dict]:
        """Paginate through Gamma API markets endpoint."""
        url = f"{GAMMA_API}/markets"
        all_markets = []
        offset = 0
        limit = int(params.get("limit", 100))

        while True:
            page_params = {**params, "offset": offset, "limit": min(limit, 100)}
            data = await self._get(url, params=page_params)

            if not data or not isinstance(data, list):
                break

            all_markets.extend(data)

            if len(data) < 100 or len(all_markets) >= limit:
                break

            offset += len(data)
            await asyncio.sleep(0.2)  # Be polite to the API

        return all_markets[:limit]

    # ─── Parsing ──────────────────────────────────────────────────────────────

    def _parse_market(self, raw: dict) -> Optional[Market]:
        """Convert raw Gamma API market dict into a typed Market object."""
        try:
            # Token IDs: Polymarket returns a list of token objects
            tokens = raw.get("tokens", [])
            token_yes = next(
                (t["token_id"] for t in tokens if t.get("outcome", "").upper() == "YES"),
                tokens[0]["token_id"] if tokens else None,
            )
            token_no = next(
                (t["token_id"] for t in tokens if t.get("outcome", "").upper() == "NO"),
                tokens[1]["token_id"] if len(tokens) > 1 else None,
            )

            if not token_yes or not token_no:
                return None

            # Parse prices from outcomePrices field (JSON string in API response)
            outcome_prices = raw.get("outcomePrices", "[]")
            if isinstance(outcome_prices, str):
                import json
                try:
                    prices = json.loads(outcome_prices)
                except Exception:
                    prices = []
            else:
                prices = outcome_prices

            yes_price = float(prices[0]) if len(prices) > 0 else 0.5
            no_price = float(prices[1]) if len(prices) > 1 else round(1 - yes_price, 6)

            # Parse created_at
            created_at = None
            if raw.get("createdAt"):
                try:
                    created_at = datetime.fromisoformat(
                        raw["createdAt"].replace("Z", "+00:00")
                    )
                except Exception:
                    pass

            close_time = None
            if raw.get("endDateIso"):
                try:
                    close_time = datetime.fromisoformat(
                        raw["endDateIso"].replace("Z", "+00:00")
                    )
                except Exception:
                    pass

            # Tags
            tags = []
            for t in raw.get("tags", []):
                if isinstance(t, dict):
                    tags.append(t.get("label", ""))
                elif isinstance(t, str):
                    tags.append(t)

            return Market(
                condition_id=raw.get("conditionId", ""),
                question_id=raw.get("questionId", raw.get("id", "")),
                token_id_yes=token_yes,
                token_id_no=token_no,
                question=raw.get("question", ""),
                description=raw.get("description"),
                resolution_criteria=raw.get("resolutionSource") or raw.get("description"),
                category=raw.get("category"),
                tags=tags,
                yes_price=yes_price,
                no_price=no_price,
                volume_24h=float(raw.get("volume24hr", 0) or 0),
                volume_total=float(raw.get("volume", 0) or 0),
                liquidity=float(raw.get("liquidity", 0) or 0),
                created_at=created_at,
                close_time=close_time,
            )

        except Exception as e:
            logger.warning(f"[Scanner] Failed to parse market: {e} | raw={str(raw)[:200]}")
            return None

    def _parse_order_book(self, token_id: str, data: dict) -> OrderBook:
        """Parse CLOB order book response."""
        bids = [(float(b["price"]), float(b["size"])) for b in data.get("bids", [])]
        asks = [(float(a["price"]), float(a["size"])) for a in data.get("asks", [])]

        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 1.0
        mid = (best_bid + best_ask) / 2 if bids and asks else 0.5
        spread = best_ask - best_bid if bids and asks else 1.0

        depth_bid = sum(p * s for p, s in bids)
        depth_ask = sum(p * s for p, s in asks)

        return OrderBook(
            token_id=token_id,
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            mid=round(mid, 6),
            spread=round(spread, 6),
            depth_bid_usdc=round(depth_bid, 2),
            depth_ask_usdc=round(depth_ask, 2),
        )

    # ─── HTTP helper ──────────────────────────────────────────────────────────

    async def _get(self, url: str, params: Optional[dict] = None) -> Optional[dict | list]:
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 429:
                    logger.warning("[Scanner] Rate limited — sleeping 5s")
                    await asyncio.sleep(5)
                    return None
                resp.raise_for_status()
                return await resp.json()
        except aiohttp.ClientError as e:
            logger.error(f"[Scanner] HTTP error {url}: {e}")
            return None