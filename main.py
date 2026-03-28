"""
OptionsDesk Pro — FastAPI Backend
Real-time options analytics server.

Run: uvicorn main:app --reload --port 8000

Endpoints:
  GET  /api/quote/{symbol}
  GET  /api/expiries/{symbol}
  GET  /api/chain/{symbol}/{expiry}
  GET  /api/history/{symbol}?tf=1M
  GET  /api/volatility/{symbol}
  GET  /api/vix
  POST /api/strategy/analyze
  POST /api/strategy/scenario
  WS   /ws/{symbol}           (live price + chain streaming)
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from market_data import (
    MarketDataService,
    VolatilitySurface,
    bs_greeks,
    bs_price,
    cache,
    get_data_service,
    implied_vol,
    vol_surface,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("optionsdesk")

ALPACA_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET", "")
RISK_FREE     = float(os.getenv("RISK_FREE_RATE", "0.0525"))

# ──────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ──────────────────────────────────────────────────────────────────────────────
class OptionLeg(BaseModel):
    type: str           # "call" | "put" | "stock"
    action: str         # "buy" | "sell"
    strike: float
    premium: float      # per-share price paid/received
    qty: int = 1        # number of contracts (100x multiplier)
    iv: Optional[float] = None    # IV % (e.g. 25.0 = 25%)
    expiry: Optional[str] = None  # YYYY-MM-DD

class StrategyRequest(BaseModel):
    symbol: str
    legs: List[OptionLeg]
    spot: Optional[float] = None
    risk_free: float = RISK_FREE
    size_multiplier: int = 1       # scale all quantities
    price_mode: str = "mid"        # "mid" | "bid" | "ask"

class ScenarioRequest(BaseModel):
    symbol: str
    legs: List[OptionLeg]
    spot: float
    iv_shift_pct: float = 0.0     # absolute IV percentage change (e.g. +5)
    price_shift_pct: float = 0.0  # % price move
    days_forward: int = 0
    risk_free: float = RISK_FREE

# ──────────────────────────────────────────────────────────────────────────────
# WEBSOCKET CONNECTION MANAGER
# ──────────────────────────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, symbol: str, ws: WebSocket):
        await ws.accept()
        self.connections.setdefault(symbol, []).append(ws)
        logger.info(f"[WS] Client connected: {symbol} ({len(self.connections[symbol])} total)")

    def disconnect(self, symbol: str, ws: WebSocket):
        if symbol in self.connections:
            self.connections[symbol] = [w for w in self.connections[symbol] if w != ws]
            if not self.connections[symbol]:
                del self.connections[symbol]

    async def broadcast(self, symbol: str, msg: Dict):
        dead = []
        for ws in self.connections.get(symbol, []):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(symbol, ws)

    @property
    def active_symbols(self) -> List[str]:
        return list(self.connections.keys())


manager = ConnectionManager()

# ──────────────────────────────────────────────────────────────────────────────
# BACKGROUND PRICE STREAMER
# ──────────────────────────────────────────────────────────────────────────────
async def price_streamer():
    """Background task: push live quotes to WebSocket subscribers every 5s."""
    while True:
        try:
            for symbol in manager.active_symbols:
                ds = get_data_service(ALPACA_KEY, ALPACA_SECRET)
                quote = await ds.get_quote(symbol)
                if quote:
                    await manager.broadcast(symbol, {
                        "type": "quote",
                        "symbol": symbol,
                        "price": quote["price"],
                        "change": quote["change"],
                        "change_pct": quote["change_pct"],
                        "bid": quote.get("bid"),
                        "ask": quote.get("ask"),
                        "volume": quote.get("volume"),
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": quote.get("source", "yahoo"),
                    })
        except Exception as e:
            logger.error(f"[Streamer] Error: {e}")
        await asyncio.sleep(5)


# ──────────────────────────────────────────────────────────────────────────────
# APP LIFECYCLE
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("OptionsDesk Pro backend starting…")
    asyncio.create_task(price_streamer())
    yield
    logger.info("OptionsDesk Pro backend shutting down.")


app = FastAPI(
    title="OptionsDesk Pro API",
    version="2.0.0",
    description="Real-time options analytics backend",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: DATA SERVICE
# ──────────────────────────────────────────────────────────────────────────────
def ds() -> MarketDataService:
    return get_data_service(ALPACA_KEY, ALPACA_SECRET)


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES — MARKET DATA
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/quote/{symbol}")
async def get_quote(symbol: str):
    symbol = symbol.upper().strip()
    result = await ds().get_quote(symbol)
    if not result:
        raise HTTPException(404, f"Quote not found for {symbol}")
    return result


@app.get("/api/expiries/{symbol}")
async def get_expiries(symbol: str):
    symbol = symbol.upper().strip()
    expiries = await ds().get_expiries(symbol)
    if not expiries:
        raise HTTPException(404, f"No options expiries found for {symbol}")
    return {"symbol": symbol, "expiries": expiries, "count": len(expiries)}


@app.get("/api/chain/{symbol}/{expiry}")
async def get_chain(symbol: str, expiry: str):
    symbol = symbol.upper().strip()
    chain  = await ds().get_option_chain(symbol, expiry)
    if not chain:
        raise HTTPException(404, f"Options chain not available for {symbol} {expiry}")
    return chain


@app.get("/api/history/{symbol}")
async def get_history(symbol: str, tf: str = Query("1M")):
    symbol = symbol.upper().strip()
    hist   = await ds().get_history(symbol, tf)
    if not hist:
        raise HTTPException(404, f"History not available for {symbol}")
    return hist


@app.get("/api/vix")
async def get_vix():
    vix = await ds().get_vix()
    return {"vix": vix, "timestamp": datetime.utcnow().isoformat()}


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES — VOLATILITY ANALYTICS
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/volatility/{symbol}")
async def get_volatility(symbol: str, expiry: Optional[str] = Query(None)):
    symbol = symbol.upper().strip()

    # 1. Get current quote for spot
    quote = await ds().get_quote(symbol)
    if not quote:
        raise HTTPException(404, f"Quote not found for {symbol}")
    spot = quote["price"]

    # 2. Get all expiries
    expiries = await ds().get_expiries(symbol)
    if not expiries:
        raise HTTPException(404, f"No expiries for {symbol}")

    # 3. Fetch first 6 expiries for term structure
    term_data = []
    smile_data = None
    chains_for_surface = []

    for exp in expiries[:6]:
        chain = await ds().get_option_chain(symbol, exp)
        if not chain:
            continue
        chains_for_surface.append(chain)

        # ATM IV from this expiry
        atm_iv = _get_atm_iv(chain, spot)
        if atm_iv:
            term_data.append({
                "expiry": exp,
                "dte": chain["dte"],
                "atm_iv": round(atm_iv, 2),
            })

        # Smile for selected expiry
        if exp == (expiry or expiries[0]):
            smile_data = _build_smile(chain, spot)

    # 4. Build vol surface
    if chains_for_surface:
        vol_surface.build(symbol, chains_for_surface, spot)

    # 5. Historical vol + IV rank
    hist_result = await ds().get_history(symbol, "1Y")
    hv30 = hist_result.get("hv30") if hist_result else None

    current_iv = term_data[0]["atm_iv"] if term_data else None
    iv_rank_data = None
    if current_iv:
        iv_rank_data = await ds().get_iv_rank(symbol, current_iv)

    # 6. Skew metrics from near-expiry chain
    skew_data = None
    if chains_for_surface:
        skew_data = _compute_skew(chains_for_surface[0], spot)

    return {
        "symbol": symbol,
        "spot": spot,
        "term_structure": term_data,
        "smile": smile_data,
        "hv30": hv30,
        "iv_rank": iv_rank_data,
        "skew": skew_data,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _get_atm_iv(chain: Dict, spot: float) -> Optional[float]:
    """Return IV of the closest-to-ATM call (or average call+put)."""
    all_options = chain.get("calls", []) + chain.get("puts", [])
    atm_opts = [o for o in all_options if o.get("iv") and o.get("open_interest", 0) > 0]
    if not atm_opts:
        return None
    closest = min(atm_opts, key=lambda o: abs(o["strike"] - spot))
    return closest.get("iv")


def _build_smile(chain: Dict, spot: float) -> Dict:
    """Build IV smile data from a single expiry chain."""
    smile = {}
    for opt_type in ("calls", "puts"):
        for c in chain.get(opt_type, []):
            k = c["strike"]
            iv = c.get("iv")
            oi = c.get("open_interest", 0)
            if iv and oi >= 1 and 0.7 * spot <= k <= 1.3 * spot:
                if k not in smile:
                    smile[k] = {"ivs": [], "oi": 0, "vol": 0}
                smile[k]["ivs"].append(iv)
                smile[k]["oi"] += oi
                smile[k]["vol"] += c.get("volume", 0)

    strikes, ivs, oi_list = [], [], []
    for k in sorted(smile.keys()):
        ivs_raw = smile[k]["ivs"]
        avg_iv  = sum(ivs_raw) / len(ivs_raw)
        strikes.append(round(k, 2))
        ivs.append(round(avg_iv, 2))
        oi_list.append(smile[k]["oi"])

    return {"strikes": strikes, "ivs": ivs, "open_interests": oi_list}


def _compute_skew(chain: Dict, spot: float) -> Dict:
    """Compute 25-delta put/call skew and risk reversal."""
    calls = {c["strike"]: c for c in chain.get("calls", []) if c.get("iv")}
    puts  = {p["strike"]: p for p in chain.get("puts", [])  if p.get("iv")}

    # Find 25-delta options
    put25 = min(puts.values(), key=lambda p: abs(abs(p["delta"]) - 0.25), default=None) if puts else None
    call25 = min(calls.values(), key=lambda c: abs(c["delta"] - 0.25), default=None) if calls else None
    atm_call = min(calls.values(), key=lambda c: abs(c["strike"] - spot), default=None) if calls else None

    skew = {}
    if put25 and call25:
        rr25 = round(put25["iv"] - call25["iv"], 2)
        skew["risk_reversal_25d"] = rr25
        skew["put_25d_iv"]  = put25["iv"]
        skew["call_25d_iv"] = call25["iv"]
    if atm_call:
        skew["atm_iv"] = atm_call["iv"]
        if put25:
            skew["put_skew"] = round(put25["iv"] - atm_call["iv"], 2) if atm_call else None
        if call25:
            skew["call_skew"] = round(call25["iv"] - atm_call["iv"], 2) if atm_call else None
    return skew


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES — STRATEGY ANALYTICS
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/strategy/analyze")
async def analyze_strategy(req: StrategyRequest):
    """
    Full strategy analysis: payoff curve, Greeks, breakevens,
    max profit/loss, using real per-contract IV and market prices.
    """
    symbol = req.symbol.upper()

    # Get live spot if not supplied
    if req.spot is None:
        quote = await ds().get_quote(symbol)
        spot = quote["price"] if quote else 500.0
    else:
        spot = req.spot

    legs = req.legs
    if not legs:
        return {"error": "No legs provided"}

    size = req.size_multiplier
    r = req.risk_free

    # Compute T for each leg
    leg_data = []
    for leg in legs:
        if leg.expiry:
            exp_dt = datetime.strptime(leg.expiry, "%Y-%m-%d")
            T = max((exp_dt - datetime.utcnow()).days, 0) / 365.0
        else:
            T = 30 / 365.0  # default 30 DTE

        iv = (leg.iv or 25.0) / 100.0  # Convert % to decimal

        leg_data.append({
            "type": leg.type,
            "action": leg.action,
            "strike": leg.strike,
            "premium": leg.premium,
            "qty": leg.qty * size,
            "iv": iv,
            "T": T,
            "expiry": leg.expiry,
        })

    # Build payoff curve
    spread = max(spot * 0.15, 60)
    prices = np.linspace(spot - spread, spot + spread, 200)

    expiry_pnl = []
    current_pnl = []
    greek_curves = {"delta": [], "gamma": [], "theta": [], "vega": []}
    T_current = sum(l["T"] for l in leg_data if l["type"] != "stock") / max(len([l for l in leg_data if l["type"] != "stock"]), 1) * 0.5

    for Sp in prices:
        exp_pl = _calc_expiry_pnl(leg_data, Sp)
        cur_pl = _calc_current_pnl(leg_data, Sp, r)
        expiry_pnl.append(round(float(exp_pl), 2))
        current_pnl.append(round(float(cur_pl), 2))

        # Greeks at each price point
        g = _calc_portfolio_greeks(leg_data, Sp, T_current, r)
        for gk in greek_curves:
            greek_curves[gk].append(round(float(g.get(gk, 0)), 6))

    # Portfolio Greeks at spot
    portfolio_greeks = _calc_portfolio_greeks(leg_data, spot, leg_data[0]["T"] if leg_data else 30/365, r)

    # Stats
    exp_arr = np.array(expiry_pnl)
    max_profit = float(np.max(exp_arr))
    max_loss   = float(np.min(exp_arr))
    profit_pts = int(np.sum(exp_arr > 0))
    win_pct    = round(profit_pts / len(exp_arr) * 100, 1)

    # Breakevens
    breakevens = []
    for i in range(1, len(expiry_pnl)):
        if (expiry_pnl[i-1] < 0 and expiry_pnl[i] >= 0) or \
           (expiry_pnl[i-1] >= 0 and expiry_pnl[i] < 0):
            # Linear interpolation
            frac = -expiry_pnl[i-1] / (expiry_pnl[i] - expiry_pnl[i-1] + 1e-9)
            be = (spot - spread) + (2*spread) * (i-1+frac) / len(prices)
            breakevens.append(round(be, 2))

    # Net cost / credit
    net_cost = sum(
        (1 if l["action"] == "buy" else -1) * l["qty"] * 100 * l["premium"]
        for l in leg_data if l["type"] != "stock"
    )

    # Expected move (1-SD based on ATM IV)
    atm_iv = _get_portfolio_atm_iv(leg_data, spot)
    T_ref  = leg_data[0]["T"] if leg_data else 30/365
    expected_move = round(spot * atm_iv * np.sqrt(T_ref), 2)

    # Risk metrics
    daily_move = spot * atm_iv / np.sqrt(252)
    var_95 = round(1.645 * daily_move * abs(portfolio_greeks.get("delta", 0)), 2)

    return {
        "symbol": symbol,
        "spot": spot,
        "prices": [round(float(p), 2) for p in prices],
        "expiry_pnl": expiry_pnl,
        "current_pnl": current_pnl,
        "greek_curves": greek_curves,
        "portfolio_greeks": {k: round(float(v), 6) for k, v in portfolio_greeks.items()},
        "stats": {
            "max_profit": round(max_profit, 2) if max_profit < 1e6 else None,
            "max_loss":   round(max_loss,   2) if max_loss   > -1e6 else None,
            "win_pct":    win_pct,
            "breakevens": breakevens,
            "net_cost":   round(net_cost, 2),
            "expected_move": expected_move,
            "var_95": var_95,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/strategy/scenario")
async def scenario_analysis(req: ScenarioRequest):
    """What-if scenario: shift IV, price, time."""
    spot_new = req.spot * (1 + req.price_shift_pct / 100)
    r = req.risk_free

    leg_data = []
    for leg in req.legs:
        if leg.expiry:
            exp_dt = datetime.strptime(leg.expiry, "%Y-%m-%d")
            T_orig = max((exp_dt - datetime.utcnow()).days, 0) / 365.0
        else:
            T_orig = 30 / 365.0
        T_new = max(T_orig - req.days_forward / 365.0, 0.001)
        iv_new = ((leg.iv or 25.0) + req.iv_shift_pct) / 100.0

        leg_data.append({
            "type": leg.type,
            "action": leg.action,
            "strike": leg.strike,
            "premium": leg.premium,
            "qty": leg.qty,
            "iv": iv_new,
            "T": T_new,
        })

    pnl = _calc_current_pnl(leg_data, spot_new, r)
    greeks = _calc_portfolio_greeks(leg_data, spot_new, leg_data[0]["T"] if leg_data else 0.08, r)

    return {
        "scenario_pnl": round(float(pnl), 2),
        "new_spot": round(spot_new, 4),
        "greeks": {k: round(float(v), 6) for k, v in greeks.items()},
    }


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY MATH HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _calc_expiry_pnl(legs: List[Dict], Sp: float) -> float:
    total = 0.0
    for leg in legs:
        sign = 1 if leg["action"] == "buy" else -1
        if leg["type"] == "stock":
            total += sign * leg["qty"] * (Sp - leg["premium"])
        else:
            intrinsic = max(Sp - leg["strike"], 0) if leg["type"] == "call" \
                        else max(leg["strike"] - Sp, 0)
            total += sign * leg["qty"] * 100 * (intrinsic - leg["premium"])
    return total


def _calc_current_pnl(legs: List[Dict], Sp: float, r: float) -> float:
    total = 0.0
    for leg in legs:
        sign = 1 if leg["action"] == "buy" else -1
        if leg["type"] == "stock":
            total += sign * leg["qty"] * (Sp - leg["premium"])
        else:
            T = max(leg.get("T", 0.08), 1e-4)
            iv = leg.get("iv", 0.25)
            curr_price = bs_price(Sp, leg["strike"], T, r, iv, leg["type"])
            total += sign * leg["qty"] * 100 * (curr_price - leg["premium"])
    return total


def _calc_portfolio_greeks(legs: List[Dict], S: float, T: float, r: float) -> Dict:
    result = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    for leg in legs:
        sign = 1 if leg["action"] == "buy" else -1
        if leg["type"] == "stock":
            result["delta"] += sign * leg["qty"]
            continue
        T_leg = max(leg.get("T", T), 1e-4)
        iv    = leg.get("iv", 0.25)
        g = bs_greeks(S, leg["strike"], T_leg, r, iv, leg["type"])
        mult = sign * leg["qty"] * 100
        for gk in result:
            result[gk] += g.get(gk, 0) * mult
    return result


def _get_portfolio_atm_iv(legs: List[Dict], spot: float) -> float:
    ivs = [l["iv"] for l in legs if l["type"] != "stock" and l.get("iv")]
    return sum(ivs) / len(ivs) if ivs else 0.20


# ──────────────────────────────────────────────────────────────────────────────
# WEBSOCKET ENDPOINT
# ──────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    symbol = symbol.upper().strip()
    await manager.connect(symbol, websocket)

    # Send initial snapshot
    try:
        quote = await ds().get_quote(symbol)
        if quote:
            await websocket.send_json({"type": "snapshot", **quote})

        async for msg in websocket.iter_json():
            # Handle client messages (e.g., subscribe to different symbol)
            action = msg.get("action")
            if action == "ping":
                await websocket.send_json({"type": "pong", "ts": datetime.utcnow().isoformat()})
            elif action == "subscribe":
                new_sym = msg.get("symbol", symbol).upper()
                if new_sym != symbol:
                    manager.disconnect(symbol, websocket)
                    await manager.connect(new_sym, websocket)
                    symbol = new_sym

    except WebSocketDisconnect:
        manager.disconnect(symbol, websocket)
        logger.info(f"[WS] Client disconnected: {symbol}")
    except Exception as e:
        logger.error(f"[WS] Error for {symbol}: {e}")
        manager.disconnect(symbol, websocket)


# ──────────────────────────────────────────────────────────────────────────────
# HEALTH + STATUS
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "alpaca_enabled": bool(ALPACA_KEY),
        "active_ws_symbols": manager.active_symbols,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/cache/flush")
async def flush_cache():
    cache.flush()
    return {"status": "cache flushed"}


# ──────────────────────────────────────────────────────────────────────────────
# DEV ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True,
                log_level="info", access_log=True)
