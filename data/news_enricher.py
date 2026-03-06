"""
news_enricher.py — fetches relevant news for a market and produces a
condensed context string ready for LLM consumption.

Sources (in priority order):
  1. NewsAPI.org   — structured, keyword search, free tier: 100 req/day
  2. RSS feeds     — unlimited, covers major outlets
  3. GDELT         — massive dataset, free, no key required
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp
import feedparser
from loguru import logger

from core.config import config
from core.models import EnrichedMarket, Market, NewsItem


# ─── RSS feeds to scrape as free fallback ────────────────────────────────────
RSS_FEEDS = [
    ("Reuters", "https://feeds.reuters.com/reuters/topNews"),
    ("BBC", "http://feeds.bbci.co.uk/news/rss.xml"),
    ("AP News", "https://feeds.apnews.com/rss/apf-topnews"),
    ("The Guardian", "https://www.theguardian.com/world/rss"),
    ("NPR", "https://feeds.npr.org/1001/rss.xml"),
    ("FT", "https://www.ft.com/rss/home"),
    ("Bloomberg Politics", "https://feeds.bloomberg.com/politics/news.rss"),
    ("Politico", "https://rss.politico.com/politics-news.xml"),
    ("ESPN", "https://www.espn.com/espn/rss/news"),
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
]

# Stop words to strip from keyword extraction
STOP_WORDS = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were",
    "have", "has", "had", "do", "does", "did", "this", "that",
    "with", "for", "from", "or", "in", "on", "at", "to", "of",
    "and", "by", "it", "its", "who", "what", "when", "where", "how",
    "than", "more", "most", "least", "first", "last", "next",
}


class NewsEnricher:
    """
    Given a Market object, fetches relevant recent news and returns
    an EnrichedMarket with a condensed news_summary string.
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._rss_cache: dict[str, list[NewsItem]] = {}  # feed_url → items
        self._rss_last_fetch: dict[str, datetime] = {}

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={"User-Agent": "Mozilla/5.0 (polymarket-scanner)"},
        )
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    # ─── Public interface ─────────────────────────────────────────────────────

    async def enrich(self, market: Market) -> EnrichedMarket:
        """Fetch relevant news for a market and produce enriched context."""
        keywords = self._extract_keywords(market.question)
        logger.debug(f"[News] Enriching '{market.question[:60]}' | keywords: {keywords}")

        # Run news sources concurrently
        tasks = [
            self._fetch_newsapi(keywords),
            self._fetch_rss(keywords),
            self._fetch_gdelt(keywords),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_items: list[NewsItem] = []
        for r in results:
            if isinstance(r, list):
                all_items.extend(r)
            elif isinstance(r, Exception):
                logger.debug(f"[News] Source failed: {r}")

        # Score relevance and deduplicate
        scored = self._score_and_deduplicate(all_items, keywords)

        # Keep top 8 most relevant items
        top_items = scored[:8]

        summary = self._build_summary(market, top_items)

        return EnrichedMarket(
            market=market,
            news_items=top_items,
            news_summary=summary,
        )

    async def enrich_batch(self, markets: list[Market]) -> list[EnrichedMarket]:
        """Enrich multiple markets concurrently with rate limit respect."""
        sem = asyncio.Semaphore(5)  # Limit concurrent news requests

        async def enrich_one(m: Market) -> EnrichedMarket:
            async with sem:
                return await self.enrich(m)

        return list(await asyncio.gather(*[enrich_one(m) for m in markets]))

    # ─── News sources ─────────────────────────────────────────────────────────

    async def _fetch_newsapi(self, keywords: list[str]) -> list[NewsItem]:
        """NewsAPI.org — structured search, best quality."""
        if not config.news.newsapi_enabled:
            return []

        query = " OR ".join(f'"{k}"' if " " in k else k for k in keywords[:5])
        from_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "relevancy",
            "pageSize": 10,
            "apiKey": config.news.newsapi_key,
            "language": "en",
        }

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 429:
                    logger.warning("[News] NewsAPI rate limited")
                    return []
                data = await resp.json()

            items = []
            for a in data.get("articles", []):
                published = None
                if a.get("publishedAt"):
                    try:
                        published = datetime.fromisoformat(
                            a["publishedAt"].replace("Z", "+00:00")
                        )
                    except Exception:
                        pass
                items.append(NewsItem(
                    title=a.get("title", ""),
                    description=a.get("description"),
                    url=a.get("url", ""),
                    published_at=published,
                    source=a.get("source", {}).get("name", "NewsAPI"),
                ))
            return items

        except Exception as e:
            logger.debug(f"[News] NewsAPI error: {e}")
            return []

    async def _fetch_rss(self, keywords: list[str]) -> list[NewsItem]:
        """
        Fetch and cache RSS feeds, then filter for relevant articles.
        RSS is free/unlimited — good fallback when NewsAPI quota is exhausted.
        """
        # Refresh stale caches (older than 15 minutes)
        feeds_to_refresh = []
        for name, url in RSS_FEEDS:
            last = self._rss_last_fetch.get(url)
            if not last or (datetime.now(timezone.utc) - last).seconds > 900:
                feeds_to_refresh.append((name, url))

        if feeds_to_refresh:
            await self._refresh_rss_caches(feeds_to_refresh)

        # Search cached items for keyword matches
        keyword_set = set(k.lower() for k in keywords)
        relevant: list[NewsItem] = []

        for _, url in RSS_FEEDS:
            for item in self._rss_cache.get(url, []):
                text = f"{item.title} {item.description or ''}".lower()
                hits = sum(1 for kw in keyword_set if kw in text)
                if hits > 0:
                    item.relevance_score = hits / len(keyword_set)
                    relevant.append(item)

        return relevant

    async def _refresh_rss_caches(self, feeds: list[tuple[str, str]]):
        """Fetch multiple RSS feeds concurrently."""
        async def fetch_feed(name: str, url: str):
            try:
                async with self._session.get(url) as resp:
                    content = await resp.text()
                feed = feedparser.parse(content)
                items = []
                for entry in feed.entries[:20]:
                    published = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        try:
                            import time as _time
                            published = datetime.fromtimestamp(
                                _time.mktime(entry.published_parsed), tz=timezone.utc
                            )
                        except Exception:
                            pass
                    items.append(NewsItem(
                        title=getattr(entry, "title", ""),
                        description=getattr(entry, "summary", None),
                        url=getattr(entry, "link", ""),
                        published_at=published,
                        source=name,
                    ))
                self._rss_cache[url] = items
                self._rss_last_fetch[url] = datetime.now(timezone.utc)
            except Exception as e:
                logger.debug(f"[News] RSS fetch failed {name}: {e}")

        await asyncio.gather(*[fetch_feed(n, u) for n, u in feeds])

    async def _fetch_gdelt(self, keywords: list[str]) -> list[NewsItem]:
        """
        GDELT GKG API — completely free, no key, massive coverage.
        Returns articles mentioning key themes from last 24 hours.
        """
        query = "%20".join(keywords[:3])
        url = (
            f"https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={query}&mode=artlist&maxrecords=10"
            f"&format=json&timespan=1d&sort=DateDesc"
        )

        try:
            async with self._session.get(url) as resp:
                data = await resp.json(content_type=None)

            items = []
            for a in (data or {}).get("articles", []):
                items.append(NewsItem(
                    title=a.get("title", ""),
                    description=None,
                    url=a.get("url", ""),
                    source="GDELT/" + a.get("domain", ""),
                ))
            return items

        except Exception as e:
            logger.debug(f"[News] GDELT error: {e}")
            return []

    # ─── Utilities ────────────────────────────────────────────────────────────

    def _extract_keywords(self, question: str) -> list[str]:
        """
        Extract meaningful keywords from a market question.
        E.g. "Will Joe Biden win the 2024 election?" → ["Joe Biden", "2024 election"]
        """
        # Extract quoted phrases first
        quoted = re.findall(r'"([^"]+)"', question)

        # Extract capitalised proper nouns (names, places, orgs)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)

        # Extract year patterns
        years = re.findall(r'\b(20\d{2})\b', question)

        # Individual non-stop words
        words = [
            w.lower() for w in re.findall(r'\b\w+\b', question)
            if w.lower() not in STOP_WORDS and len(w) > 3
        ]

        # Combine, deduplicate, prioritise longer phrases
        keywords = []
        seen = set()

        for phrase in quoted + proper_nouns:
            if phrase.lower() not in seen and len(phrase) > 3:
                keywords.append(phrase)
                seen.add(phrase.lower())

        for w in years + words:
            if w.lower() not in seen:
                keywords.append(w)
                seen.add(w.lower())

        return keywords[:10]

    def _score_and_deduplicate(
        self, items: list[NewsItem], keywords: list[str]
    ) -> list[NewsItem]:
        """Score items by keyword relevance, remove URL duplicates, sort."""
        keyword_set = set(k.lower() for k in keywords)
        seen_urls: set[str] = set()
        scored: list[NewsItem] = []

        for item in items:
            if item.url in seen_urls or not item.title:
                continue
            seen_urls.add(item.url)

            text = f"{item.title} {item.description or ''}".lower()
            hits = sum(1 for kw in keyword_set if kw in text)
            item.relevance_score = hits / max(len(keyword_set), 1)
            scored.append(item)

        return sorted(scored, key=lambda x: (x.relevance_score, x.published_at or datetime.min), reverse=True)

    def _build_summary(self, market: Market, items: list[NewsItem]) -> str:
        """Build a concise text summary of news for LLM consumption."""
        if not items:
            return "No recent relevant news found."

        lines = [f"Market: {market.question}"]
        if market.resolution_criteria:
            lines.append(f"Resolution criteria: {market.resolution_criteria[:300]}")
        lines.append(f"\nRecent news ({len(items)} articles):")

        for i, item in enumerate(items, 1):
            date_str = item.published_at.strftime("%Y-%m-%d") if item.published_at else "recent"
            desc = f" — {item.description[:150]}" if item.description else ""
            lines.append(f"  {i}. [{date_str}] {item.title}{desc}")

        return "\n".join(lines)