"""
OptionsDesk Pro — Market Data Layer
Integrates: Yahoo Finance (yfinance), Alpaca, Polygon (free tier)
Handles: quotes, options chains, OHLCV history, real IV computation
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm

logger = logging.getLogger("optionsdesk.data")

# ──────────────────────────────────────────────────────────────────────────────
# IN-MEMORY CACHE  (Redis-compatible interface; swap out easily)
# ──────────────────────────────────────────────────────────────────────────────
class MemoryCache:
    def __init__(self):
        self._store: Dict[str, Tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self._store:
            val, exp = self._store[key]
            if time.time() < exp:
                return val
            del self._store[key]
        return None

    def set(self, key: str, val: Any, ttl: int = 60):
        self._store[key] = (val, time.time() + ttl)

    def delete(self, key: str):
        self._store.pop(key, None)

    def flush(self):
        self._store.clear()

try:
    import redis.asyncio as aioredis
    _redis: Optional[aioredis.Redis] = None
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

cache = MemoryCache()


# ──────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES + IV SOLVER
# ──────────────────────────────────────────────────────────────────────────────
def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:
    if T <= 1e-6 or sigma <= 1e-6:
        return max(S - K, 0) if opt_type == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "call":
        return max(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0)
    return max(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> Dict:
    if T <= 1e-6 or sigma <= 1e-6:
        return {
            "delta": (1.0 if S > K else 0.0) if opt_type == "call" else (-1.0 if S < K else 0.0),
            "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0
        }
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf_d1 = norm.pdf(d1)
    if opt_type == "call":
        delta = norm.cdf(d1)
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    gamma = pdf_d1 / (S * sigma * sqrtT)
    vega  = S * pdf_d1 * sqrtT / 100
    theta = (-(S * pdf_d1 * sigma) / (2 * sqrtT)
             - r * K * np.exp(-r * T) * (norm.cdf(d2) if opt_type == "call" else norm.cdf(-d2))) / 365
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                opt_type: str, tol: float = 1e-6) -> Optional[float]:
    """Newton-Raphson / Brent hybrid IV solver."""
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    intrinsic = max(S - K, 0) if opt_type == "call" else max(K - S, 0)
    if market_price < intrinsic - 0.01:
        return None

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, opt_type) - market_price

    try:
        # Fast Newton-Raphson first
        sigma = 0.30
        for _ in range(50):
            px = bs_price(S, K, T, r, sigma, opt_type)
            g  = bs_greeks(S, K, T, r, sigma, opt_type)
            vega = g["vega"] * 100  # un-normalise
            if abs(vega) < 1e-10:
                break
            sigma -= (px - market_price) / vega
            sigma = max(0.001, min(sigma, 20.0))
            if abs(px - market_price) < tol:
                return round(sigma, 6)
        # Fallback Brent
        result = brentq(objective, 0.001, 15.0, xtol=tol, maxiter=200)
        return round(result, 6)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# YAHOO FINANCE DATA LAYER
# ──────────────────────────────────────────────────────────────────────────────
class YahooDataLayer:
    """Primary free data source. Uses yfinance library."""

    RISK_FREE_RATE = 0.0525  # ~5.25% US T-bill, update periodically

    async def get_quote(self, symbol: str) -> Optional[Dict]:
        key = f"quote:{symbol}"
        cached = cache.get(key)
        if cached:
            return cached

        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            info = await asyncio.to_thread(lambda: ticker.info)
            fast_info = await asyncio.to_thread(lambda: ticker.fast_info)

            price = (
                getattr(fast_info, "last_price", None)
                or info.get("regularMarketPrice")
                or info.get("currentPrice")
            )
            if not price:
                return None

            prev_close = (
                getattr(fast_info, "previous_close", None)
                or info.get("regularMarketPreviousClose")
                or info.get("previousClose")
                or price
            )

            result = {
                "symbol": symbol,
                "price": round(float(price), 4),
                "prev_close": round(float(prev_close), 4),
                "change": round(float(price) - float(prev_close), 4),
                "change_pct": round((float(price) - float(prev_close)) / float(prev_close) * 100, 4),
                "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
                "day_low": info.get("dayLow") or info.get("regularMarketDayLow"),
                "week52_high": info.get("fiftyTwoWeekHigh"),
                "week52_low": info.get("fiftyTwoWeekLow"),
                "volume": info.get("regularMarketVolume") or info.get("volume"),
                "avg_volume": info.get("averageVolume"),
                "market_cap": info.get("marketCap"),
                "beta": info.get("beta"),
                "short_name": info.get("shortName", symbol),
                "timestamp": datetime.utcnow().isoformat(),
            }
            cache.set(key, result, ttl=15)
            return result

        except Exception as e:
            logger.warning(f"[Yahoo] Quote failed for {symbol}: {e}")
            return None

    async def get_expiries(self, symbol: str) -> List[str]:
        key = f"expiries:{symbol}"
        cached = cache.get(key)
        if cached:
            return cached
        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            expiries = await asyncio.to_thread(lambda: ticker.options)
            result = list(expiries) if expiries else []
            cache.set(key, result, ttl=300)
            return result
        except Exception as e:
            logger.warning(f"[Yahoo] Expiries failed for {symbol}: {e}")
            return []

    async def get_option_chain(self, symbol: str, expiry: str,
                               spot: float, r: float = None) -> Optional[Dict]:
        key = f"chain:{symbol}:{expiry}"
        cached = cache.get(key)
        if cached:
            return cached

        r = r or self.RISK_FREE_RATE
        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            chain  = await asyncio.to_thread(lambda: ticker.option_chain(expiry))
            calls_df = chain.calls
            puts_df  = chain.puts

            expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
            T = max((expiry_dt - datetime.utcnow()).days, 0) / 365.0

            def process_contracts(df, opt_type: str) -> List[Dict]:
                contracts = []
                for _, row in df.iterrows():
                    K = float(row.get("strike", 0))
                    if K <= 0:
                        continue

                    bid  = float(row.get("bid") or 0)
                    ask  = float(row.get("ask") or 0)
                    last = float(row.get("lastPrice") or 0)
                    mid  = round((bid + ask) / 2, 4) if bid > 0 and ask > 0 else last
                    vol  = int(row.get("volume") or 0)
                    oi   = int(row.get("openInterest") or 0)
                    itm  = bool(row.get("inTheMoney", False))

                    # Use Yahoo's IV if present; else compute from mid price
                    yf_iv = row.get("impliedVolatility")
                    iv_val = float(yf_iv) if yf_iv and float(yf_iv) > 0.001 else None

                    # Always recompute IV from mid for accuracy (Yahoo IV can be stale)
                    price_for_iv = mid if mid > 0 else last
                    if price_for_iv > 0 and T > 0 and spot > 0:
                        computed_iv = implied_vol(price_for_iv, spot, K, T, r, opt_type)
                        if computed_iv and 0.001 < computed_iv < 15.0:
                            iv_val = computed_iv

                    # Compute Greeks using real per-contract IV
                    if iv_val and T > 0:
                        greeks = bs_greeks(spot, K, T, r, iv_val, opt_type)
                    else:
                        iv_val = iv_val or 0.25  # last-resort fallback
                        greeks = bs_greeks(spot, K, T, r, iv_val, opt_type)

                    # Bid-ask spread
                    spread = round(ask - bid, 4) if ask > 0 and bid > 0 else None
                    spread_pct = round(spread / mid * 100, 2) if spread and mid > 0 else None

                    contracts.append({
                        "strike": K,
                        "type": opt_type,
                        "bid": round(bid, 4),
                        "ask": round(ask, 4),
                        "mid": round(mid, 4),
                        "last": round(last, 4),
                        "price": round(mid if mid > 0 else last, 4),
                        "volume": vol,
                        "open_interest": oi,
                        "iv": round(iv_val * 100, 4) if iv_val else None,  # as pct
                        "iv_raw": iv_val,
                        "itm": itm,
                        "spread": spread,
                        "spread_pct": spread_pct,
                        "delta": round(greeks["delta"], 6),
                        "gamma": round(greeks["gamma"], 8),
                        "theta": round(greeks["theta"], 6),
                        "vega": round(greeks["vega"], 6),
                        "rho": round(greeks["rho"], 6),
                        "dte": max((expiry_dt - datetime.utcnow()).days, 0),
                        "expiry": expiry,
                        # Liquidity score: 0-100
                        "liquidity_score": _liquidity_score(vol, oi, spread_pct),
                    })
                return sorted(contracts, key=lambda x: x["strike"])

            result = {
                "symbol": symbol,
                "expiry": expiry,
                "dte": max((expiry_dt - datetime.utcnow()).days, 0),
                "spot": spot,
                "calls": process_contracts(calls_df, "call"),
                "puts": process_contracts(puts_df, "put"),
                "fetched_at": datetime.utcnow().isoformat(),
                "source": "yahoo_finance",
            }
            cache.set(key, result, ttl=60)
            return result

        except Exception as e:
            logger.warning(f"[Yahoo] Chain failed for {symbol} {expiry}: {e}")
            return None

    async def get_history(self, symbol: str, period: str, interval: str) -> Optional[Dict]:
        key = f"history:{symbol}:{period}:{interval}"
        cached = cache.get(key)
        if cached:
            return cached
        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            hist   = await asyncio.to_thread(lambda: ticker.history(period=period, interval=interval))
            if hist.empty:
                return None

            candles, volumes = [], []
            for idx, row in hist.iterrows():
                ts = int(idx.timestamp())
                if None in (row.Open, row.High, row.Low, row.Close):
                    continue
                candles.append({
                    "time": ts,
                    "open":  round(float(row.Open), 4),
                    "high":  round(float(row.High), 4),
                    "low":   round(float(row.Low), 4),
                    "close": round(float(row.Close), 4),
                })
                volumes.append({
                    "time": ts,
                    "value": int(row.Volume) if row.Volume else 0,
                    "color": "rgba(38,166,154,0.5)" if float(row.Close) >= float(row.Open)
                             else "rgba(239,83,80,0.5)"
                })

            # Historical volatility (30-day annualised)
            closes = [c["close"] for c in candles[-31:]]
            hv30 = None
            if len(closes) >= 10:
                rets = np.diff(np.log(closes))
                hv30 = round(float(np.std(rets) * np.sqrt(252) * 100), 2)

            result = {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "candles": candles,
                "volumes": volumes,
                "hv30": hv30,
            }
            cache.set(key, result, ttl=60 if interval in ("1m", "5m", "15m") else 300)
            return result

        except Exception as e:
            logger.warning(f"[Yahoo] History failed {symbol}: {e}")
            return None

    async def get_vix(self) -> Optional[float]:
        key = "vix:latest"
        cached = cache.get(key)
        if cached:
            return cached
        try:
            ticker = await asyncio.to_thread(yf.Ticker, "^VIX")
            info   = await asyncio.to_thread(lambda: ticker.fast_info)
            vix    = getattr(info, "last_price", None)
            if vix:
                vix = round(float(vix), 2)
                cache.set(key, vix, ttl=30)
            return vix
        except Exception:
            return None


def _liquidity_score(volume: int, oi: int, spread_pct: Optional[float]) -> int:
    """0–100 liquidity score based on volume, OI, spread."""
    score = 0
    if oi > 10000:  score += 35
    elif oi > 1000: score += 25
    elif oi > 100:  score += 15
    elif oi > 10:   score += 5

    if volume > 5000:  score += 35
    elif volume > 500: score += 25
    elif volume > 50:  score += 15
    elif volume > 5:   score += 5

    if spread_pct is not None:
        if spread_pct < 2:   score += 30
        elif spread_pct < 5:  score += 20
        elif spread_pct < 10: score += 10
    return min(score, 100)


# ──────────────────────────────────────────────────────────────────────────────
# ALPACA DATA LAYER  (real-time quotes — paper or live key)
# ──────────────────────────────────────────────────────────────────────────────
class AlpacaDataLayer:
    """Alpaca v2 market data API — free tier, real-time US equity quotes."""

    BASE_URL = "https://data.alpaca.markets/v2"

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.enabled    = bool(api_key and api_secret)

    @property
    def _headers(self) -> Dict:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }

    async def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        if not self.enabled:
            return None
        key = f"alpaca:quote:{symbol}"
        cached = cache.get(key)
        if cached:
            return cached
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    f"{self.BASE_URL}/stocks/{symbol}/quotes/latest",
                    headers=self._headers
                )
                if r.status_code == 200:
                    data = r.json()
                    q = data.get("quote", {})
                    result = {
                        "bid": q.get("bp"),
                        "ask": q.get("ap"),
                        "bid_size": q.get("bs"),
                        "ask_size": q.get("as"),
                        "timestamp": q.get("t"),
                        "source": "alpaca",
                    }
                    cache.set(key, result, ttl=5)
                    return result
        except Exception as e:
            logger.debug(f"[Alpaca] Quote failed for {symbol}: {e}")
        return None

    async def get_latest_trade(self, symbol: str) -> Optional[float]:
        if not self.enabled:
            return None
        key = f"alpaca:trade:{symbol}"
        cached = cache.get(key)
        if cached:
            return cached
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    f"{self.BASE_URL}/stocks/{symbol}/trades/latest",
                    headers=self._headers
                )
                if r.status_code == 200:
                    price = r.json().get("trade", {}).get("p")
                    if price:
                        price = float(price)
                        cache.set(key, price, ttl=5)
                        return price
        except Exception as e:
            logger.debug(f"[Alpaca] Trade failed for {symbol}: {e}")
        return None

    async def get_bars(self, symbol: str, timeframe: str = "1Day",
                       start: str = None, end: str = None, limit: int = 200) -> Optional[List[Dict]]:
        if not self.enabled:
            return None
        if not start:
            start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
        key = f"alpaca:bars:{symbol}:{timeframe}:{start}"
        cached = cache.get(key)
        if cached:
            return cached
        try:
            params = {"timeframe": timeframe, "start": start, "limit": limit, "adjustment": "all"}
            if end:
                params["end"] = end
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    f"{self.BASE_URL}/stocks/{symbol}/bars",
                    params=params, headers=self._headers
                )
                if r.status_code == 200:
                    bars = r.json().get("bars", [])
                    result = [{
                        "time": int(datetime.fromisoformat(b["t"].replace("Z","")).timestamp()),
                        "open":  round(b["o"], 4),
                        "high":  round(b["h"], 4),
                        "low":   round(b["l"], 4),
                        "close": round(b["c"], 4),
                        "volume": b.get("v", 0),
                    } for b in bars]
                    cache.set(key, result, ttl=300)
                    return result
        except Exception as e:
            logger.debug(f"[Alpaca] Bars failed for {symbol}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# VOLATILITY SURFACE
# ──────────────────────────────────────────────────────────────────────────────
class VolatilitySurface:
    """
    Builds a real vol surface from options chain data.
    Supports per-strike IV, skew, term structure, IV Rank/Percentile.
    Uses scipy RBF interpolation across (strike, dte) space.
    """

    def __init__(self):
        self._surfaces: Dict[str, Dict] = {}

    def build(self, symbol: str, chains: List[Dict], spot: float):
        """Build surface from list of chains across expiries."""
        strikes, dtes, ivs = [], [], []

        for chain in chains:
            dte = chain.get("dte", 30)
            for opt_type in ("calls", "puts"):
                for c in chain.get(opt_type, []):
                    if c.get("iv_raw") and c.get("open_interest", 0) >= 10:
                        K = c["strike"]
                        iv = c["iv_raw"]
                        moneyness = K / spot
                        if 0.7 < moneyness < 1.3:  # Only near-money options
                            strikes.append(K)
                            dtes.append(dte)
                            ivs.append(iv)

        if len(ivs) < 4:
            return None

        try:
            from scipy.interpolate import RBFInterpolator
            points = np.column_stack([strikes, dtes])
            values = np.array(ivs)
            rbf = RBFInterpolator(points, values, smoothing=0.1, kernel="thin_plate_spline")
            self._surfaces[symbol] = {
                "interpolator": rbf,
                "strikes": strikes,
                "dtes": dtes,
                "ivs": ivs,
                "spot": spot,
                "built_at": time.time(),
            }
            return True
        except Exception as e:
            logger.warning(f"[VolSurface] Build failed for {symbol}: {e}")
            return None

    def get_iv(self, symbol: str, strike: float, dte: float) -> Optional[float]:
        """Interpolate IV for any (strike, dte) pair."""
        surf = self._surfaces.get(symbol)
        if not surf:
            return None
        if time.time() - surf["built_at"] > 300:
            return None
        try:
            iv = float(surf["interpolator"]([[strike, dte]])[0])
            return max(0.01, min(iv, 5.0))
        except Exception:
            return None

    def get_smile(self, symbol: str, dte: float, spot: float,
                  n_strikes: int = 30) -> Optional[Dict]:
        """Return the IV smile for a given expiry (list of strike → IV pairs)."""
        surf = self._surfaces.get(symbol)
        if not surf:
            return None
        try:
            k_min = spot * 0.80
            k_max = spot * 1.20
            strike_range = np.linspace(k_min, k_max, n_strikes)
            pts = np.column_stack([strike_range, np.full(n_strikes, dte)])
            ivs = surf["interpolator"](pts)
            ivs = np.clip(ivs, 0.01, 5.0)
            return {
                "strikes": [round(float(k), 2) for k in strike_range],
                "ivs": [round(float(iv) * 100, 2) for iv in ivs],
                "moneyness": [round(float(k) / spot, 4) for k in strike_range],
            }
        except Exception as e:
            logger.warning(f"[VolSurface] Smile failed: {e}")
            return None

    def iv_rank(self, symbol: str, current_iv: float,
                history_ivs: List[float]) -> Optional[Dict]:
        """Compute IV Rank and IV Percentile from historical data."""
        if not history_ivs or len(history_ivs) < 10:
            return None
        hi = max(history_ivs)
        lo = min(history_ivs)
        iv_rank = round((current_iv - lo) / (hi - lo) * 100, 1) if hi != lo else 50.0
        iv_pct  = round(sum(1 for v in history_ivs if v < current_iv) / len(history_ivs) * 100, 1)
        return {
            "iv_rank": iv_rank,
            "iv_percentile": iv_pct,
            "iv_52w_high": round(hi * 100, 2),
            "iv_52w_low":  round(lo * 100, 2),
            "current_iv":  round(current_iv * 100, 2),
        }


vol_surface = VolatilitySurface()


# ──────────────────────────────────────────────────────────────────────────────
# UNIFIED DATA SERVICE
# ──────────────────────────────────────────────────────────────────────────────
class MarketDataService:
    """
    Single entry point for all market data.
    Priority: Alpaca (real-time price) → Yahoo Finance (chains, history)
    """

    def __init__(self, alpaca_key: str = "", alpaca_secret: str = ""):
        self.yahoo  = YahooDataLayer()
        self.alpaca = AlpacaDataLayer(alpaca_key, alpaca_secret)

    async def get_quote(self, symbol: str) -> Optional[Dict]:
        # Try Alpaca first for real-time price
        alpaca_trade = await self.alpaca.get_latest_trade(symbol)
        alpaca_quote = await self.alpaca.get_latest_quote(symbol)

        # Always get Yahoo for metadata
        yq = await self.yahoo.get_quote(symbol)

        if yq:
            # Overlay Alpaca's more recent price if available
            if alpaca_trade:
                yq["price"] = alpaca_trade
                yq["change"] = round(alpaca_trade - yq["prev_close"], 4)
                yq["change_pct"] = round(yq["change"] / yq["prev_close"] * 100, 4)
                yq["source"] = "alpaca+yahoo"
            if alpaca_quote:
                yq["bid"] = alpaca_quote.get("bid")
                yq["ask"] = alpaca_quote.get("ask")
            return yq
        return None

    async def get_option_chain(self, symbol: str, expiry: str) -> Optional[Dict]:
        quote = await self.get_quote(symbol)
        spot  = quote["price"] if quote else None
        if not spot:
            return None
        return await self.yahoo.get_option_chain(symbol, expiry, spot)

    async def get_expiries(self, symbol: str) -> List[str]:
        return await self.yahoo.get_expiries(symbol)

    async def get_history(self, symbol: str, timeframe: str = "1M") -> Optional[Dict]:
        tf_map = {
            "1D":  ("1d",  "5m"),
            "5D":  ("5d",  "15m"),
            "1M":  ("1mo", "1d"),
            "3M":  ("3mo", "1d"),
            "6M":  ("6mo", "1wk"),
            "1Y":  ("1y",  "1wk"),
            "2Y":  ("2y",  "1mo"),
        }
        period, interval = tf_map.get(timeframe, ("1mo", "1d"))

        # Try Alpaca bars first (cleaner data)
        alp_tf = {"1D":"1Min", "5D":"5Min", "1M":"1Day", "3M":"1Day", "6M":"1Week", "1Y":"1Day"}
        alpaca_bars = await self.alpaca.get_bars(symbol, timeframe=alp_tf.get(timeframe, "1Day"))
        if alpaca_bars and len(alpaca_bars) > 5:
            closes = [b["close"] for b in alpaca_bars[-31:]]
            hv30 = None
            if len(closes) >= 10:
                rets = np.diff(np.log(closes))
                hv30 = round(float(np.std(rets) * np.sqrt(252) * 100), 2)

            # Build volume colors
            volumes = [{
                "time": b["time"],
                "value": b.get("volume", 0),
                "color": "rgba(38,166,154,0.5)" if b["close"] >= b["open"] else "rgba(239,83,80,0.5)"
            } for b in alpaca_bars]

            return {
                "symbol": symbol, "period": period, "interval": interval,
                "candles": alpaca_bars, "volumes": volumes, "hv30": hv30, "source": "alpaca"
            }

        # Fallback Yahoo
        return await self.yahoo.get_history(symbol, period, interval)

    async def get_vix(self) -> Optional[float]:
        return await self.yahoo.get_vix()

    async def build_vol_surface(self, symbol: str) -> bool:
        """Fetch multiple expiries and build vol surface."""
        quote    = await self.get_quote(symbol)
        if not quote:
            return False
        spot     = quote["price"]
        expiries = await self.get_expiries(symbol)
        if not expiries:
            return False

        chains = []
        # Fetch first 6 expiries for surface
        for exp in expiries[:6]:
            chain = await self.get_option_chain(symbol, exp)
            if chain:
                chains.append(chain)

        if chains:
            vol_surface.build(symbol, chains, spot)
            return True
        return False

    async def get_iv_rank(self, symbol: str, current_iv: float) -> Optional[Dict]:
        """Compute IV Rank / IV Percentile using 252-day history."""
        hist = await self.get_history(symbol, "1Y")
        if not hist:
            return None
        candles = hist.get("candles", [])
        if len(candles) < 30:
            return None

        # Compute rolling 30-day HV for each window as proxy for historical IV
        closes = np.array([c["close"] for c in candles])
        hist_ivs = []
        for i in range(30, len(closes)):
            window = closes[i-30:i]
            rets   = np.diff(np.log(window))
            hv     = float(np.std(rets) * np.sqrt(252))
            hist_ivs.append(hv)

        return vol_surface.iv_rank(symbol, current_iv / 100, hist_ivs)


# Singleton
_data_service: Optional[MarketDataService] = None

def get_data_service(alpaca_key: str = "", alpaca_secret: str = "") -> MarketDataService:
    global _data_service
    if not _data_service:
        _data_service = MarketDataService(alpaca_key, alpaca_secret)
    return _data_service
