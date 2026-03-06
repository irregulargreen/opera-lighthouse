"""OANDA REST API adapter for market data and account info."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import config
from core.models import Candle, PriceSeries


GRANULARITY_MAP = {
    "M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30",
    "H1": "H1", "H4": "H4", "D": "D", "W": "W", "M": "M",
}


class OANDAClient:
    """Async client for OANDA v20 REST API."""

    def __init__(self):
        self._base_url = config.oanda.base_url
        self._headers = {
            "Authorization": f"Bearer {config.oanda.api_key}",
            "Content-Type": "application/json",
        }
        self._account_id = config.oanda.account_id

    # ── Market Data ──────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def get_candles(
        self,
        instrument: str,
        granularity: str = "H1",
        count: int = 500,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
    ) -> PriceSeries:
        """Fetch historical candlestick data."""
        params: dict = {
            "granularity": GRANULARITY_MAP.get(granularity, granularity),
            "price": "M",  # mid prices
        }
        if from_time:
            params["from"] = from_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            if to_time:
                params["to"] = to_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                params["count"] = min(count, 5000)
        else:
            params["count"] = min(count, 5000)

        url = f"{self._base_url}/v3/instruments/{instrument}/candles"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=self._headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        candles = []
        for c in data.get("candles", []):
            if not c.get("complete", True):
                continue
            mid = c.get("mid", {})
            candles.append(Candle(
                timestamp=self._parse_oanda_time(c["time"]),
                open=float(mid.get("o", 0)),
                high=float(mid.get("h", 0)),
                low=float(mid.get("l", 0)),
                close=float(mid.get("c", 0)),
                volume=float(c.get("volume", 0)),
            ))

        return PriceSeries(instrument=instrument, timeframe=granularity, candles=candles)

    async def get_current_price(self, instrument: str) -> dict:
        """Get current bid/ask/mid price."""
        url = f"{self._base_url}/v3/accounts/{self._account_id}/pricing"
        params = {"instruments": instrument}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=self._headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        prices = data.get("prices", [])
        if not prices:
            return {}

        p = prices[0]
        bid = float(p["bids"][0]["price"]) if p.get("bids") else 0.0
        ask = float(p["asks"][0]["price"]) if p.get("asks") else 0.0
        return {
            "instrument": instrument,
            "bid": bid,
            "ask": ask,
            "mid": round((bid + ask) / 2, 6),
            "spread": round(ask - bid, 6),
            "time": p.get("time"),
        }

    async def get_multiple_timeframes(
        self,
        instrument: str,
        timeframes: list[str],
        count: int = 500,
    ) -> dict[str, PriceSeries]:
        """Fetch candles for multiple timeframes concurrently."""
        tasks = [
            self.get_candles(instrument, tf, count)
            for tf in timeframes
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[str, PriceSeries] = {}
        for tf, result in zip(timeframes, results):
            if isinstance(result, PriceSeries):
                out[tf] = result
            else:
                logger.warning(f"[Data] Failed to fetch {instrument} {tf}: {result}")
        return out

    # ── Account ──────────────────────────────────────────────────────────────

    async def get_account_summary(self) -> dict:
        """Get account balance, margin, open positions."""
        url = f"{self._base_url}/v3/accounts/{self._account_id}/summary"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()

        acct = data.get("account", {})
        return {
            "balance": float(acct.get("balance", 0)),
            "nav": float(acct.get("NAV", 0)),
            "unrealized_pnl": float(acct.get("unrealizedPL", 0)),
            "margin_used": float(acct.get("marginUsed", 0)),
            "margin_available": float(acct.get("marginAvailable", 0)),
            "open_trade_count": int(acct.get("openTradeCount", 0)),
            "currency": acct.get("currency", "USD"),
        }

    async def get_open_trades(self) -> list[dict]:
        """Get all currently open trades."""
        url = f"{self._base_url}/v3/accounts/{self._account_id}/openTrades"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()

        return data.get("trades", [])

    # ── Orders ───────────────────────────────────────────────────────────────

    async def place_market_order(
        self,
        instrument: str,
        units: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> dict:
        """Place a market order with optional SL/TP."""
        order: dict = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(int(units)),
            "timeInForce": "FOK",
        }
        if stop_loss is not None:
            order["stopLossOnFill"] = {"price": f"{stop_loss:.5f}"}
        if take_profit is not None:
            order["takeProfitOnFill"] = {"price": f"{take_profit:.5f}"}

        url = f"{self._base_url}/v3/accounts/{self._account_id}/orders"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url, headers=self._headers, json={"order": order}
            )
            resp.raise_for_status()
            data = resp.json()

        if "orderFillTransaction" in data:
            fill = data["orderFillTransaction"]
            logger.info(
                f"[OANDA] Filled: {instrument} {units} units @ {fill.get('price')}"
            )
        elif "orderCancelTransaction" in data:
            reason = data["orderCancelTransaction"].get("reason", "unknown")
            logger.warning(f"[OANDA] Order cancelled: {reason}")

        return data

    async def close_trade(self, trade_id: str, units: Optional[int] = None) -> dict:
        """Close an open trade (fully or partially)."""
        url = f"{self._base_url}/v3/accounts/{self._account_id}/trades/{trade_id}/close"
        body = {}
        if units is not None:
            body["units"] = str(units)

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.put(url, headers=self._headers, json=body)
            resp.raise_for_status()
            return resp.json()

    async def modify_trade(
        self,
        trade_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop_distance: Optional[float] = None,
    ) -> dict:
        """Modify SL/TP on an open trade."""
        body: dict = {}
        if stop_loss is not None:
            body["stopLoss"] = {"price": f"{stop_loss:.5f}"}
        if take_profit is not None:
            body["takeProfit"] = {"price": f"{take_profit:.5f}"}
        if trailing_stop_distance is not None:
            body["trailingStopLoss"] = {"distance": f"{trailing_stop_distance:.5f}"}

        url = f"{self._base_url}/v3/accounts/{self._account_id}/trades/{trade_id}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.put(url, headers=self._headers, json=body)
            resp.raise_for_status()
            return resp.json()

    # ── Instrument Info ──────────────────────────────────────────────────────

    async def get_instrument_info(self, instrument: str) -> dict:
        """Get pip value, min trade size, margin requirements."""
        url = f"{self._base_url}/v3/accounts/{self._account_id}/instruments"
        params = {"instruments": instrument}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=self._headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        instruments = data.get("instruments", [])
        if not instruments:
            return {}

        inst = instruments[0]
        pip_location = int(inst.get("pipLocation", -4))
        return {
            "name": inst.get("name"),
            "type": inst.get("type"),
            "pip_value": 10 ** pip_location,
            "display_precision": int(inst.get("displayPrecision", 5)),
            "min_trade_size": float(inst.get("minimumTradeSize", 1)),
            "max_trade_size": float(inst.get("maximumOrderUnits", 100000000)),
            "margin_rate": float(inst.get("marginRate", 0.02)),
        }

    @staticmethod
    def _parse_oanda_time(time_str: str) -> datetime:
        """Parse OANDA's RFC3339 timestamps which may have nanosecond precision."""
        import re
        # Trim nanoseconds down to microseconds (6 digits max)
        cleaned = re.sub(
            r"(\.\d{6})\d+",
            r"\1",
            time_str,
        )
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
