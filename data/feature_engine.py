"""Computes technical indicators and ML features from price series."""
from __future__ import annotations

import numpy as np
from loguru import logger

from core.models import PriceSeries, Regime


class FeatureEngine:
    """
    Computes all technical indicators on a PriceSeries in-place.
    Also extracts flat feature vectors for ML models.
    """

    def compute_indicators(self, series: PriceSeries) -> PriceSeries:
        """Attach all technical indicators to a PriceSeries."""
        closes = series.closes
        highs = series.highs
        lows = series.lows
        n = len(closes)

        if n < 50:
            logger.warning(f"[Features] Only {n} candles — need >=50 for indicators")
            return series

        series.sma_20 = self._sma(closes, 20)
        series.sma_50 = self._sma(closes, 50)
        series.sma_200 = self._sma(closes, min(200, n - 1))
        series.ema_12 = self._ema(closes, 12)
        series.ema_26 = self._ema(closes, 26)
        series.rsi_14 = self._rsi(closes, 14)

        macd_l, macd_s, macd_h = self._macd(closes)
        series.macd_line = macd_l
        series.macd_signal = macd_s
        series.macd_histogram = macd_h

        bb_u, bb_m, bb_l = self._bollinger(closes, 20, 2.0)
        series.bb_upper = bb_u
        series.bb_middle = bb_m
        series.bb_lower = bb_l

        series.atr_14 = self._atr(highs, lows, closes, 14)
        series.adx_14 = self._adx(highs, lows, closes, 14)

        dc_u, dc_l = self._donchian(highs, lows, 20)
        series.donchian_upper = dc_u
        series.donchian_lower = dc_l

        series.returns = self._returns(closes)
        series.volatility_20 = self._rolling_std(series.returns, 20)

        return series

    def extract_features(self, series: PriceSeries) -> dict[str, float]:
        """
        Extract a flat feature dict from a PriceSeries (for ML models).
        All features are normalised or ratio-based for model stability.
        """
        if not series.candles or len(series.candles) < 50:
            return {}

        c = series.closes
        price = c[-1]
        n = len(c)

        def _safe_last(arr, default=0.0):
            if arr and len(arr) >= n:
                v = arr[-1]
                return float(v) if v is not None and not np.isnan(v) else default
            return default

        sma20 = _safe_last(series.sma_20, price)
        sma50 = _safe_last(series.sma_50, price)
        sma200 = _safe_last(series.sma_200, price)
        ema12 = _safe_last(series.ema_12, price)
        ema26 = _safe_last(series.ema_26, price)
        rsi = _safe_last(series.rsi_14, 50.0)
        macd_hist = _safe_last(series.macd_histogram)
        atr = _safe_last(series.atr_14, price * 0.01)
        adx = _safe_last(series.adx_14, 20.0)
        bb_upper = _safe_last(series.bb_upper, price)
        bb_lower = _safe_last(series.bb_lower, price)
        bb_mid = _safe_last(series.bb_middle, price)
        vol = _safe_last(series.volatility_20, 0.01)
        dc_upper = _safe_last(series.donchian_upper, price)
        dc_lower = _safe_last(series.donchian_lower, price)

        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0
        bb_pct = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
        dc_width = (dc_upper - dc_lower) / price if price > 0 else 0
        dc_pct = (price - dc_lower) / (dc_upper - dc_lower) if (dc_upper - dc_lower) > 0 else 0.5

        # MA alignment: +1 if price > sma20 > sma50 > sma200 (strong uptrend)
        ma_alignment = 0.0
        if price > sma20:
            ma_alignment += 0.33
        if sma20 > sma50:
            ma_alignment += 0.33
        if sma50 > sma200:
            ma_alignment += 0.34
        if price < sma20 < sma50 < sma200:
            ma_alignment = -1.0
        elif price < sma20:
            ma_alignment = -(abs(ma_alignment))

        # Recent momentum (5-bar and 20-bar returns)
        ret_5 = (price - c[-6]) / c[-6] if n > 5 else 0.0
        ret_20 = (price - c[-21]) / c[-21] if n > 20 else 0.0

        # ATR ratio (current vs 50-bar average ATR) — volatility expansion/contraction
        if series.atr_14 and len(series.atr_14) >= 50:
            atr_vals = [v for v in series.atr_14[-50:] if v is not None and not np.isnan(v)]
            atr_avg = float(np.mean(atr_vals)) if atr_vals else atr
            atr_ratio = atr / atr_avg if atr_avg > 0 else 1.0
        else:
            atr_ratio = 1.0

        # Price distance from SMAs (normalised by ATR)
        dist_sma20 = (price - sma20) / atr if atr > 0 else 0
        dist_sma50 = (price - sma50) / atr if atr > 0 else 0

        return {
            "rsi": rsi,
            "adx": adx,
            "macd_histogram": macd_hist / atr if atr > 0 else 0,
            "ma_alignment": ma_alignment,
            "bb_pct": bb_pct,
            "bb_width": bb_width,
            "dc_pct": dc_pct,
            "dc_width": dc_width,
            "atr_ratio": atr_ratio,
            "volatility": vol,
            "dist_sma20": dist_sma20,
            "dist_sma50": dist_sma50,
            "ret_5": ret_5,
            "ret_20": ret_20,
            "volume_ratio": self._volume_ratio(series),
        }

    def classify_regime(self, features: dict[str, float]) -> Regime:
        """
        Rule-based regime classification (used before ML regime model is trained).
        """
        adx = features.get("adx", 20)
        ma_align = features.get("ma_alignment", 0)
        atr_ratio = features.get("atr_ratio", 1.0)
        bb_width = features.get("bb_width", 0)

        # High volatility override
        if atr_ratio > 1.8 or bb_width > 0.06:
            return Regime.VOLATILE

        # Strong trend
        if adx > 30 and abs(ma_align) > 0.6:
            return Regime.TRENDING_UP if ma_align > 0 else Regime.TRENDING_DOWN

        # Moderate trend
        if adx > 20 and abs(ma_align) > 0.3:
            return Regime.TRENDING_UP if ma_align > 0 else Regime.TRENDING_DOWN

        return Regime.RANGING

    # ── Indicator implementations ────────────────────────────────────────────

    @staticmethod
    def _sma(data: np.ndarray, period: int) -> list[float]:
        n = len(data)
        out = [float('nan')] * n
        if n < period:
            return out
        cumsum = np.cumsum(data)
        cumsum = np.insert(cumsum, 0, 0)
        for i in range(period - 1, n):
            out[i] = float((cumsum[i + 1] - cumsum[i + 1 - period]) / period)
        return out

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> list[float]:
        n = len(data)
        out = [float('nan')] * n
        if n < period:
            return out
        multiplier = 2.0 / (period + 1)
        out[period - 1] = float(np.mean(data[:period]))
        for i in range(period, n):
            out[i] = (data[i] - out[i - 1]) * multiplier + out[i - 1]
        return out

    @staticmethod
    def _rsi(data: np.ndarray, period: int = 14) -> list[float]:
        n = len(data)
        out = [float('nan')] * n
        if n < period + 1:
            return out
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))
        if avg_loss == 0:
            out[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[period] = 100.0 - (100.0 / (1.0 + rs))
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                out[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                out[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        return out

    def _macd(self, data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = self._ema(data, fast)
        ema_slow = self._ema(data, slow)
        n = len(data)
        macd_line = [float('nan')] * n
        for i in range(n):
            if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]):
                macd_line[i] = ema_fast[i] - ema_slow[i]
        valid = [v for v in macd_line if not np.isnan(v)]
        if len(valid) >= signal:
            signal_line = self._ema(np.array(valid), signal)
            padded_signal = [float('nan')] * (n - len(valid)) + signal_line
        else:
            padded_signal = [float('nan')] * n
        histogram = [float('nan')] * n
        for i in range(n):
            if not np.isnan(macd_line[i]) and not np.isnan(padded_signal[i]):
                histogram[i] = macd_line[i] - padded_signal[i]
        return macd_line, padded_signal, histogram

    @staticmethod
    def _bollinger(data: np.ndarray, period: int = 20, std_dev: float = 2.0):
        n = len(data)
        upper = [float('nan')] * n
        middle = [float('nan')] * n
        lower = [float('nan')] * n
        for i in range(period - 1, n):
            window = data[i - period + 1:i + 1]
            m = float(np.mean(window))
            s = float(np.std(window, ddof=1))
            middle[i] = m
            upper[i] = m + std_dev * s
            lower[i] = m - std_dev * s
        return upper, middle, lower

    @staticmethod
    def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14):
        n = len(closes)
        out = [float('nan')] * n
        if n < period + 1:
            return out
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        out[period] = float(np.mean(tr[1:period + 1]))
        for i in range(period + 1, n):
            out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
        return out

    @staticmethod
    def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14):
        n = len(closes)
        out = [float('nan')] * n
        if n < 2 * period + 1:
            return out

        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        tr = np.zeros(n)

        for i in range(1, n):
            up = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]
            plus_dm[i] = up if up > down and up > 0 else 0
            minus_dm[i] = down if down > up and down > 0 else 0
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

        atr_val = float(np.mean(tr[1:period + 1]))
        plus_di_val = float(np.mean(plus_dm[1:period + 1]))
        minus_di_val = float(np.mean(minus_dm[1:period + 1]))

        dx_values = []
        for i in range(period + 1, n):
            atr_val = (atr_val * (period - 1) + tr[i]) / period
            plus_di_val = (plus_di_val * (period - 1) + plus_dm[i]) / period
            minus_di_val = (minus_di_val * (period - 1) + minus_dm[i]) / period

            if atr_val > 0:
                pdi = 100 * plus_di_val / atr_val
                mdi = 100 * minus_di_val / atr_val
            else:
                pdi = mdi = 0

            di_sum = pdi + mdi
            dx = 100 * abs(pdi - mdi) / di_sum if di_sum > 0 else 0
            dx_values.append(dx)

            if len(dx_values) == period:
                out[i] = float(np.mean(dx_values))
            elif len(dx_values) > period:
                out[i] = (out[i - 1] * (period - 1) + dx) / period

        return out

    @staticmethod
    def _donchian(highs: np.ndarray, lows: np.ndarray, period: int = 20):
        n = len(highs)
        upper = [float('nan')] * n
        lower = [float('nan')] * n
        for i in range(period - 1, n):
            upper[i] = float(np.max(highs[i - period + 1:i + 1]))
            lower[i] = float(np.min(lows[i - period + 1:i + 1]))
        return upper, lower

    @staticmethod
    def _returns(data: np.ndarray) -> list[float]:
        n = len(data)
        out = [0.0] + [float((data[i] - data[i - 1]) / data[i - 1]) if data[i - 1] != 0 else 0.0 for i in range(1, n)]
        return out

    @staticmethod
    def _rolling_std(data: list[float], period: int) -> list[float]:
        n = len(data)
        out = [float('nan')] * n
        arr = np.array(data)
        for i in range(period - 1, n):
            out[i] = float(np.std(arr[i - period + 1:i + 1], ddof=1))
        return out

    @staticmethod
    def _volume_ratio(series: PriceSeries) -> float:
        vols = series.volumes
        if len(vols) < 21:
            return 1.0
        recent = float(np.mean(vols[-5:]))
        avg = float(np.mean(vols[-20:]))
        return recent / avg if avg > 0 else 1.0
