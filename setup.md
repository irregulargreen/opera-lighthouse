# Setup Guide — Polymarket AI Bot

Complete checklist to go from zero to running, in order.

---

## What you need before starting

### 1. Accounts and wallets

**Polymarket account (required for trading)**
- Go to polymarket.com and create an account
- Complete KYC if using the US platform (required post Dec 2025)
- Fund your account with USDC on Polygon network
- Two wallet options:
  - **MetaMask / hardware wallet** → `POLYMARKET_SIGNATURE_TYPE=0` (EOA)
  - **Magic.link email wallet** → `POLYMARKET_SIGNATURE_TYPE=1` (proxy wallet)
- You need two addresses: your **private key** (never share) and your **funder address** (the proxy wallet Polymarket deploys for you, visible in account settings)

**To find your funder address:**
Log into Polymarket → Profile → Settings → scroll to "Proxy Wallet Address"

**Minimum recommended capital:** $200–500 USDC to start
With MAX_TRADE_SIZE_USDC=$50 and up to 10 positions, you need enough buffer.

### 2. LLM API keys (need at least one)

**Option A — Anthropic Claude (recommended)**
- Sign up at console.anthropic.com
- Create API key → copy to `ANTHROPIC_API_KEY`
- The bot uses `claude-haiku-4-5` for cheap screening and `claude-sonnet-4` for final analysis

**Option B — OpenAI GPT**
- Sign up at platform.openai.com
- Create API key → copy to `OPENAI_API_KEY`
- Uses `gpt-4o-mini` for screening, `gpt-4o` for final analysis

**Option C — Ollama (free, runs locally)**
- Install from https://ollama.ai
- Run: `ollama serve` (keep running in background)
- Pull a model: `ollama pull llama3.1:8b`
- No key needed — just set `OLLAMA_BASE_URL=http://localhost:11434`

**Recommendation:** Run all three. Ollama does free Tier 1 screening.
Cloud APIs only hit ~10% of markets. Monthly cost: ~$30-50.

### 3. News API key (optional but improves edge)

- Sign up at newsapi.org → free tier: 100 requests/day
- Copy key to `NEWSAPI_KEY`
- If not set, bot uses RSS feeds + GDELT (free, unlimited) as fallback

### 4. Python environment

```bash
python3 --version   # Need 3.10+
```

---

## Installation

```bash
# 1. Clone / download the project
cd polymarket_bot

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Verify Ollama setup (if using local LLMs)
python ollama_setup.py
```

---

## Configuration

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
# Required for trading
POLYMARKET_PRIVATE_KEY=0x...        # Your wallet private key
POLYMARKET_FUNDER_ADDRESS=0x...     # Your Polymarket proxy wallet address
POLYMARKET_SIGNATURE_TYPE=0         # 0 for MetaMask, 1 for Magic.link

# LLMs — at least one required
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
OLLAMA_MODEL=llama3.1:8b

# News (optional)
NEWSAPI_KEY=...

# Safety — START WITH THIS TRUE
DRY_RUN=true

# Sizing — start conservative
TOTAL_CAPITAL_USDC=500.0
MAX_TRADE_SIZE_USDC=25.0
MIN_EDGE=0.06
KELLY_FRACTION=0.25
```

---

## Bootstrap the ML models (do this once before first run)

The ML market classifier works best when seeded with historical data.
Without this it falls back to heuristic rules, which still works but is less sharp.

```bash
python3 -c "
import asyncio
from utils.ml_market_classifier import bootstrap_from_historical, MarketClassifier
clf = MarketClassifier()
asyncio.run(bootstrap_from_historical(clf, n_markets=1000))
"
```

This fetches ~1000 resolved Polymarket markets and trains the initial classifier.
Takes about 2-3 minutes. Needs no API keys (uses public Gamma API).

---

## Test run (no trading)

```bash
# Verify everything is wired up — scan markets and show top signals
python analyse.py --limit 50 --verbose

# Check only new markets (highest edge window)
python analyse.py --new-only

# Search a specific category
python analyse.py --query "bitcoin"

# Show reasoning for the top signal
python analyse.py --reasoning --limit 30
```

If you see a table of signals with edges and reasoning, everything is working.

---

## Paper trading phase (run for 2-3 weeks minimum)

```bash
# DRY_RUN=true in .env (default)
python main.py
```

The bot will:
- Scan for new markets every 2 minutes
- Do a full sweep every 15 minutes
- Log all signals and simulated trades to `logs/calibration.db`
- Print cost summary per session

Check progress at any time:
```bash
python analyse.py --stats
```

**When to switch to live trading:**
- 30+ resolved trades in calibration.db
- Brier score < 0.20 (random = 0.25)
- Simulated P&L is positive
- Win rate > 55%

The meta-learner needs 100 resolved trades before its signal is reliable.
The Platt calibration scaler needs 30.
Both fall back to conservative heuristics until then.

---

## Going live

```bash
# In .env:
DRY_RUN=false
MAX_TRADE_SIZE_USDC=25.0   # Keep small until you trust the model

python main.py
```

---

## Monitoring

```bash
# View performance stats and calibration curve
python analyse.py --stats

# Watch live logs
tail -f logs/bot.log        # If you redirect stdout

# Check the SQLite database directly
sqlite3 logs/calibration.db "
  SELECT market_question, side, market_price, our_probability,
         edge_after_fees, pnl_usdc, brier_score
  FROM trades
  WHERE resolved=1
  ORDER BY executed_at DESC
  LIMIT 20;"
```

---

## File structure at runtime

```
polymarket_bot/
├── main.py              ← Run this for continuous trading
├── analyse.py           ← Run this for one-off analysis
├── ollama_setup.py      ← Run once to check/benchmark Ollama
├── .env                 ← Your config (never commit this)
├── requirements.txt
│
├── logs/                ← Auto-created on first run
│   ├── calibration.db           SQLite: all trades + outcomes
│   ├── calibration_models.json  Platt scaler training data
│   ├── market_classifier.pkl    Gradient boosting model (market screener)
│   ├── meta_learner.pkl         Meta-learner model
│   └── meta_training_data.json  Meta-learner training data
│
├── core/
│   ├── config.py        ← Settings (reads .env)
│   └── models.py        ← Shared data types
│
├── data/
│   ├── scanner.py       ← Gamma API + CLOB order books
│   └── news_enricher.py ← NewsAPI + RSS + GDELT
│
├── strategies/
│   ├── tiered_forecaster.py  ← 3-tier LLM pipeline (THE MAIN BRAIN)
│   ├── edge_calculator.py    ← Kelly sizing + signal filtering
│   └── executor.py           ← py-clob-client order placement
│
└── utils/
    ├── calibration.py          ← SQLite trade tracker + Brier scores
    ├── ml_calibration.py       ← Platt scaling (LLM bias correction)
    ├── ml_market_classifier.py ← Market pre-filter (gradient boosting)
    └── meta_learner.py         ← Final gate + adaptive Kelly (THE LEARNER)
```

---

## Full data flow (one trade cycle)

```
Every 2 min:  Gamma API → new markets
Every 15 min: Gamma API → active markets ($1k–$500k liquidity sweet spot)
                ↓
              CLOB API → live order books (concurrent, 20 at a time)
                ↓
    [FREE]  ML market classifier → score each market for mispricing likelihood
              Skip if score < 0.25  (cuts ~70% of markets before any LLM call)
                ↓
    [FREE]  NewsAPI + RSS + GDELT → recent news for shortlisted markets
                ↓
    [FREE]  Tier 1: Ollama local LLM → fast screen, skip if edge < 6%
                ↓
    [CHEAP] Tier 2: Claude Haiku / GPT-4o-mini → validate, skip if edge < 5%
                ↓
    [FULL]  Tier 3: Claude Sonnet + GPT-4o → ensemble estimate
              Platt scaling calibration → shrinkage → crowd blend
                ↓
              Edge calculator → Kelly sizing → signal generated
                ↓
    [GATE]  Meta-learner → P(this trade actually wins | all signals)
              Adaptive Kelly multiplier → adjust size up/down
              Skip if meta-learner not confident
                ↓
              Executor → limit order via py-clob-client
                ↓
              calibration.db → record trade
                ↓
    [LATER] Market resolves → bot.resolve_trade(order_id, won, pnl)
              → ML Calibrator updated → Platt scaler refit
              → Meta-learner updated → model refit every 25 trades
              → Market classifier updated → model refit every 50 trades
```

---

## Resolving trades (wiring the feedback loop)

The bot doesn't auto-detect resolution yet (that's a next step).
For now, manually call resolve when you see a market settle:

```python
from main import PolymarketBot
bot = PolymarketBot()

# After a market resolves:
bot.resolve_trade(
    order_id="your-order-id",
    won=True,        # Did our side win?
    pnl=12.50        # Actual P&L in USDC
)
```

To auto-resolve: poll `GET /data-api.polymarket.com/positions` for your wallet
and compare current position values against entry prices. This is the logical
next feature to build.

---

## Cost summary

| Component | Cost | Notes |
|-----------|------|-------|
| Polymarket trading fees | ~2% taker on crypto, 0% most markets | Accounted for in edge calc |
| OpenAI GPT-4o | ~$0-30/month | Only hits ~10% of markets (Tier 3) |
| Claude Sonnet | ~$0-30/month | Same — Tier 3 only |
| Claude Haiku | ~$2-5/month | Tier 2 validation |
| GPT-4o-mini | ~$1-3/month | Tier 2 validation |
| Ollama | $0 | Tier 1 screen + any local inference |
| NewsAPI | $0 | Free tier 100 req/day (enough) |
| GDELT + RSS | $0 | Unlimited |
| VPS (optional) | $5-20/month | Digital Ocean / Hetzner if running 24/7 |
| **Total** | **~$30-80/month** | With all providers, mostly Ollama screening |

---

## Common issues

**"No LLM providers configured"**
→ At least one of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or Ollama must be set

**"Failed to initialise CLOB client"**
→ Check `POLYMARKET_PRIVATE_KEY` starts with `0x` and is correct length (66 chars)
→ Check `POLYMARKET_FUNDER_ADDRESS` is your proxy wallet, not your main wallet

**"Rate limited"**
→ Scanner backs off automatically, but if persistent: increase `SCAN_INTERVAL_SECONDS`

**"ML pre-filter: 0 markets proceeding to LLM"**
→ Lower `min_liquidity` threshold or run `bootstrap_from_historical()` to train classifier

**Brier score not improving after 50+ trades**
→ Check `logs/calibration.db` — are all trades resolving correctly?
→ Try increasing `MIN_LLM_CONFIDENCE` to only trade on high-confidence estimates
→ Check `python analyse.py --stats` for per-category breakdown