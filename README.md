# OptionsDesk Pro v2 — Real-Data Options Analytics Platform

A professional-grade options analytics platform with a **FastAPI Python backend** and a full-featured browser frontend. All analytics are driven by real market data — no theoretical assumptions, no constant IV, no synthetic chains.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    BROWSER FRONTEND                      │
│  index.html — Vanilla JS, Chart.js, Lightweight Charts  │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  Chain   │ │Strategy  │ │ Greeks   │ │Volatility│  │
│  │  Panel   │ │ Builder  │ │ & Hedge  │ │ Surface  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                                                         │
│  Data Layer: REST API + WebSocket (live price stream)   │
│  Fallback:   Yahoo Finance direct (no backend needed)   │
└─────────────────┬───────────────────────────────────────┘
                  │ HTTP + WS
┌─────────────────▼───────────────────────────────────────┐
│               FASTAPI BACKEND  (:8000)                   │
│                                                         │
│  main.py         — Routes, WebSocket, Strategy Engine   │
│  market_data.py  — Data Layer, IV Solver, Vol Surface   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │            MarketDataService                     │   │
│  │  Priority: Alpaca (real-time) → Yahoo Finance   │   │
│  │                                                 │   │
│  │  AlpacaDataLayer     YahooDataLayer             │   │
│  │  ├─ latest_quote     ├─ quote (metadata)        │   │
│  │  ├─ latest_trade     ├─ expiries (real dates)   │   │
│  │  └─ bars (OHLCV)     ├─ option_chain            │   │
│  │                      └─ history (OHLCV)         │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  VolatilitySurface — RBF interpolation (strike × DTE)  │
│  IV Solver         — Newton-Raphson + Brent fallback    │
│  BS Greeks Engine  — Per-contract, real IV inputs       │
│  In-memory Cache   — TTL-based (swap for Redis)         │
└─────────────────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│              DATA PROVIDERS (Free)                       │
│                                                         │
│  Yahoo Finance  — Quotes, options chains, OHLCV history │
│  Alpaca Markets — Real-time equity quotes & bars        │
│  (both free tier, no paid subscription required)        │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option A — With Backend (Recommended, full real data)

```bash
# 1. Install Python 3.10+
cd optionsdesk
chmod +x start.sh

# 2. Run (Yahoo Finance only, no API keys needed)
./start.sh

# 3. With Alpaca for real-time prices (free paper account)
./start.sh --alpaca-key PKxxxxxxxx --alpaca-secret xxxxxxxxxx

# 4. Open frontend in browser
open frontend/index.html
# OR serve it:
cd frontend && python3 -m http.server 3000
# Then visit http://localhost:3000
```

### Option B — Frontend Only (No backend)
Just open `frontend/index.html` directly in any browser.
The frontend automatically falls back to fetching Yahoo Finance data directly (via CORS proxies). All analytics still work — Greeks, payoff charts, volatility, everything. Backend just adds WebSocket streaming, faster data, and Alpaca real-time prices.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Server status |
| GET | `/api/quote/{symbol}` | Live quote + metadata |
| GET | `/api/expiries/{symbol}` | All options expiry dates |
| GET | `/api/chain/{symbol}/{expiry}` | Full options chain with real IV + Greeks |
| GET | `/api/history/{symbol}?tf=1M` | OHLCV candlestick data |
| GET | `/api/volatility/{symbol}` | IV smile, term structure, IV Rank/Pct |
| POST | `/api/strategy/analyze` | Full payoff + Greeks + stats |
| POST | `/api/strategy/scenario` | What-if scenario P&L |
| WS | `/ws/{symbol}` | Live price stream |

### Example: Strategy Analysis

```bash
curl -X POST http://localhost:8000/api/strategy/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "legs": [
      {"type":"call","action":"buy","strike":585,"premium":4.50,"qty":1,"iv":19.5,"expiry":"2025-04-18"},
      {"type":"call","action":"sell","strike":590,"premium":2.20,"qty":1,"iv":18.2,"expiry":"2025-04-18"}
    ]
  }'
```

Response includes: `prices[]`, `expiry_pnl[]`, `current_pnl[]`, `greek_curves{}`, `portfolio_greeks{}`, `stats{}` — all computed from real per-contract IV.

---

## What Changed vs. v1 (Theoretical → Real Data)

| Component | Before | Now |
|-----------|--------|-----|
| **Option prices** | Black-Scholes with constant IV | Real bid/ask/mid from Yahoo Finance |
| **Implied Volatility** | Single flat IV for all strikes | Per-contract IV solved from market prices |
| **Expiry dates** | Generated algorithmically | Real expiry dates from Yahoo options API |
| **Greeks** | BS with flat IV | BS per contract using its own market IV |
| **Volatility smile** | Quadratic approximation | Real IV across all available strikes |
| **Term structure** | Random variation | Real ATM IV per expiry |
| **Vol Surface** | None | RBF interpolation over real (strike, DTE) data |
| **IV Rank/Percentile** | None | Computed from 252-day rolling HV history |
| **Spot price** | Simulated tick | Live via Alpaca + Yahoo (30s refresh + WS) |
| **Candlestick chart** | Basic layout issue (cut off) | Full LightweightCharts, fills available space |
| **Breakevens on chart** | None | Drawn as price lines directly on candle chart |

---

## Getting Alpaca API Keys (Free)

1. Go to https://alpaca.markets
2. Sign up for a **Paper Trading** account (free, no deposit)
3. Dashboard → API Keys → Generate new key
4. Use the Key ID and Secret in `start.sh` or the Config modal

Alpaca provides:
- Real-time US equity quotes (last trade price, bid/ask)
- 1-minute to daily OHLCV bars with split/dividend adjustment
- WebSocket streaming (used for live price updates)

---

## Environment Variables

```bash
ALPACA_API_KEY=PKxxxxxxxx        # Alpaca key ID (optional)
ALPACA_API_SECRET=xxxxxxxxxx     # Alpaca secret (optional)
RISK_FREE_RATE=0.0525            # Risk-free rate (auto-used in BS)
HOST=0.0.0.0                     # Bind host
PORT=8000                        # Port
```

---

## Upgrading to Redis Cache

Replace `MemoryCache` in `market_data.py`:

```python
import redis.asyncio as aioredis

redis = aioredis.from_url("redis://localhost:6379")

async def get(key): val = await redis.get(key); return json.loads(val) if val else None
async def set(key, val, ttl=60): await redis.setex(key, ttl, json.dumps(val))
```

---

## Adding PostgreSQL for Historical Storage

```python
# In main.py, add on startup:
import asyncpg

pool = await asyncpg.create_pool("postgresql://user:pass@localhost/optionsdesk")

# Store chains:
await pool.execute("""
  INSERT INTO option_chains(symbol, expiry, strike, type, iv, delta, gamma, theta, vega, bid, ask, volume, oi, fetched_at)
  VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,NOW())
  ON CONFLICT (symbol,expiry,strike,type) DO UPDATE SET iv=EXCLUDED.iv, bid=EXCLUDED.bid, ask=EXCLUDED.ask
""", symbol, expiry, strike, opt_type, iv, delta, gamma, theta, vega, bid, ask, vol, oi)
```

---

## Features Roadmap (Professional Extensions)

- [ ] **Real-time IV surface streaming** — Push vol surface updates via WebSocket
- [ ] **Order simulation** — Track fills, slippage, commission in a paper portfolio
- [ ] **Live P&L tracker** — Mark positions to market using real mid prices
- [ ] **Earnings calendar overlay** — Highlight earnings dates on candle chart + IV spikes
- [ ] **Put/Call ratio heatmap** — OI and volume across all strikes/expiries
- [ ] **Gamma exposure (GEX)** — Dealer hedging pressure by strike
- [ ] **Expected move cones** — IV-derived probability distributions on price chart
- [ ] **Multi-leg order routing** — Submit to Alpaca as combo orders
- [ ] **Backtesting engine** — Replay historical options data for strategy testing
- [ ] **Alert system** — IV rank threshold, delta threshold, P&L alerts via webhook
