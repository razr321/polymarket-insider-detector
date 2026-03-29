# Polymarket Insider Wallet Detector

Screens Polymarket wallets for patterns consistent with insider trading using six statistical signals:

| Signal | Weight | What it detects |
|--------|--------|-----------------|
| **Win Rate** | 2.5 | Binomial p-value — is the win rate statistically improbable? |
| **Wallet Age** | 2.0 | Fresh accounts (<30d) with outsized returns |
| **Pre-Resolution Timing** | 2.0 | Trades placed within 24h of market resolution |
| **PnL Concentration** | 1.5 | PnL concentrated in top 3 markets |
| **Bet Size** | 1.0 | Large average bets relative to account history |
| **Category Focus** | 1.0 | Heavy focus on politics/economics/finance |

Composite suspicion score: 0-10. Risk tiers: **CRITICAL** (>=7), **HIGH** (>=5), **MEDIUM** (>=3), **LOW** (<3).

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it Works

1. **Leaderboard scan** — Pulls top PnL wallets from Polymarket's Data API across politics, economics, and finance categories
2. **Per-wallet analysis** — For each wallet, fetches closed positions, trade history, and on-chain activity
3. **Statistical scoring** — Computes win rate significance (binomial test), wallet age, timing proximity to resolution, PnL concentration, bet sizes, and category focus
4. **Composite ranking** — Weights each signal into a 0-10 suspicion score

## Data Source

All data comes from Polymarket's **public APIs** — no authentication or API keys required:

- [Data API](https://data-api.polymarket.com) — trade history, positions, leaderboard
- [Gamma API](https://gamma-api.polymarket.com) — market metadata and categories

## CLI Usage

```bash
python detector.py  # if you want to use it as a library/script
```

```python
from detector import PolymarketClient, analyze_wallet, result_to_row

client = PolymarketClient()
result = analyze_wallet(client, "0xde7be6d489bce070a959e0cb813128ae659b5f4b")
print(f"Score: {result['score']:.1f}/10 ({result['risk']})")
```

## Streamlit Cloud

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://polymarket-insider-detector.streamlit.app)

## Disclaimer

This tool is for research and educational purposes only. A high suspicion score does not prove insider trading — it identifies statistical anomalies that warrant further investigation.
