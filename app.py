"""
Polymarket Insider Wallet Detector — Streamlit App
"""

import streamlit as st
import pandas as pd
from detector import (
    PolymarketClient, TARGET_CATEGORIES, WEIGHTS,
    scan_leaderboard, analyze_wallet, result_to_row, classify_risk,
)

st.set_page_config(
    page_title="Polymarket Insider Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Styling ──────────────────────────────────────────────────

RISK_COLORS = {
    'CRITICAL': '#ff1744',
    'HIGH': '#ff9100',
    'MEDIUM': '#ffd600',
    'LOW': '#00e676',
}

st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; }
    .risk-badge {
        padding: 2px 10px; border-radius: 4px; color: white;
        font-weight: bold; font-size: 0.85em; display: inline-block;
    }
    .metric-card {
        background: #1a1a1a; border: 1px solid #333; border-radius: 8px;
        padding: 15px; text-align: center;
    }
    .metric-value { font-size: 2em; font-weight: bold; }
    .metric-label { color: #888; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

st.title("Polymarket Insider Wallet Detector")
st.caption("Screens wallets for statistically improbable win rates, fresh accounts, "
           "concentrated bets, and pre-resolution timing across politics, economics, and finance markets.")

# ── Sidebar Controls ─────────────────────────────────────────

st.sidebar.header("Scan Settings")

mode = st.sidebar.radio("Mode", ["Leaderboard Scan", "Single Wallet"])

if mode == "Leaderboard Scan":
    categories = st.sidebar.multiselect(
        "Categories",
        ["POLITICS", "ECONOMICS", "FINANCE", "CRYPTO", "SPORTS", "CULTURE"],
        default=["POLITICS", "ECONOMICS", "FINANCE"],
    )
    top_n = st.sidebar.slider("Top N per category", 5, 100, 20)
    min_pnl = st.sidebar.number_input("Min PnL ($)", value=500, step=100)
    min_score = st.sidebar.slider("Min suspicion score", 0.0, 10.0, 0.0, 0.5)
else:
    wallet_address = st.sidebar.text_input("Wallet address", placeholder="0x...")

fast_mode = st.sidebar.checkbox("Fast mode (skip category analysis)", value=False)

run_button = st.sidebar.button("Run Scan", type="primary", use_container_width=True)

# ── Methodology ──────────────────────────────────────────────

with st.sidebar.expander("Scoring Methodology"):
    st.markdown(f"""
**Composite score (0-10) from six weighted signals:**

| Signal | Weight |
|--------|--------|
| Win Rate (binomial p-value) | {WEIGHTS['win_rate']} |
| Wallet Age | {WEIGHTS['wallet_age']} |
| Pre-Resolution Timing | {WEIGHTS['timing']} |
| PnL Concentration | {WEIGHTS['concentration']} |
| Bet Size | {WEIGHTS['bet_size']} |
| Category Focus | {WEIGHTS['category_focus']} |

**Risk tiers:** CRITICAL (>=7), HIGH (>=5), MEDIUM (>=3), LOW (<3)

**Win rate p-value:** Uses a binomial test — "what's the probability of
this win rate or better by pure chance?" Low p-value = statistically
improbable under fair/random betting.
""")

# ── Main Content ─────────────────────────────────────────────

if run_button:
    client = PolymarketClient()

    if mode == "Single Wallet":
        if not wallet_address or not wallet_address.startswith("0x"):
            st.error("Please enter a valid wallet address starting with 0x")
            st.stop()

        with st.spinner(f"Analyzing {wallet_address[:10]}..."):
            status = st.status("Analyzing wallet...", expanded=True)

            def update_status(msg):
                status.update(label=msg)

            try:
                result = analyze_wallet(
                    client, wallet_address,
                    skip_category=fast_mode,
                    progress_callback=update_status,
                )
            except Exception as e:
                st.error(f"Error analyzing wallet: {e}")
                st.stop()

            status.update(label="Analysis complete", state="complete")

        row = result_to_row(result, source_category="MANUAL")
        df = pd.DataFrame([row])

    else:
        # Leaderboard scan
        status = st.status("Scanning leaderboards...", expanded=True)

        def update_status(msg):
            status.update(label=msg)

        candidates = scan_leaderboard(client, categories, top_n, min_pnl,
                                      progress_callback=update_status)

        if not candidates:
            st.warning("No candidates found. Try lowering Min PnL or increasing Top N.")
            st.stop()

        status.update(label=f"Analyzing {len(candidates)} wallets...")
        progress_bar = st.progress(0)
        rows = []

        for i, cand in enumerate(candidates):
            progress_bar.progress((i + 1) / len(candidates),
                                  text=f"Analyzing wallet {i+1}/{len(candidates)}: "
                                       f"{cand['address'][:10]}...")
            try:
                result = analyze_wallet(
                    client, cand['address'],
                    skip_category=fast_mode,
                )
                rows.append(result_to_row(
                    result,
                    username=cand.get('username', ''),
                    source_category=cand.get('source_category', ''),
                    leaderboard_pnl=cand.get('pnl', 0),
                ))
            except Exception:
                continue

        progress_bar.empty()
        status.update(label=f"Scan complete — {len(rows)} wallets analyzed", state="complete")

        if not rows:
            st.warning("No wallets could be analyzed.")
            st.stop()

        df = pd.DataFrame(rows).sort_values('score', ascending=False)

        if min_score > 0:
            df = df[df['score'] >= min_score]

    if df.empty:
        st.info("No wallets match the current filters.")
        st.stop()

    # ── Summary Cards ────────────────────────────────────────

    col1, col2, col3, col4, col5 = st.columns(5)
    critical = len(df[df['risk'] == 'CRITICAL'])
    high = len(df[df['risk'] == 'HIGH'])
    medium = len(df[df['risk'] == 'MEDIUM'])
    low = len(df[df['risk'] == 'LOW'])

    col1.metric("CRITICAL", critical)
    col2.metric("HIGH", high)
    col3.metric("MEDIUM", medium)
    col4.metric("LOW", low)
    col5.metric("TOTAL", len(df))

    st.divider()

    # ── Results Table ────────────────────────────────────────

    st.subheader("Results")

    # Format the dataframe for display
    display_df = df[[
        'address', 'username', 'score', 'risk', 'win_rate', 'wins',
        'total_markets', 'pvalue', 'realized_pnl', 'wallet_age_days',
        'avg_bet_size', 'pre_res_pct', 'top3_pnl_share', 'insider_cat_pct',
        'source_category',
    ]].copy()

    display_df['address'] = display_df['address'].apply(
        lambda x: f"{x[:6]}...{x[-4:]}")
    display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.0%}")
    display_df['pvalue'] = display_df['pvalue'].apply(lambda x: f"{x:.4f}")
    display_df['realized_pnl'] = display_df['realized_pnl'].apply(lambda x: f"${x:,.0f}")
    display_df['avg_bet_size'] = display_df['avg_bet_size'].apply(lambda x: f"${x:,.0f}")
    display_df['pre_res_pct'] = display_df['pre_res_pct'].apply(lambda x: f"{x:.0%}")
    display_df['top3_pnl_share'] = display_df['top3_pnl_share'].apply(lambda x: f"{x:.0%}")
    display_df['insider_cat_pct'] = display_df['insider_cat_pct'].apply(lambda x: f"{x:.0%}")
    display_df['score'] = display_df['score'].apply(lambda x: f"{x:.1f}")
    display_df['wallet_age_days'] = display_df['wallet_age_days'].apply(
        lambda x: f"{x}d" if x is not None else "?")

    display_df.columns = [
        'Wallet', 'Username', 'Score', 'Risk', 'Win Rate', 'Wins',
        'Markets', 'p-value', 'Realized PnL', 'Age', 'Avg Bet',
        'Pre-Res %', 'Top-3 Conc', 'Insider Cat %', 'Source',
    ]

    def color_risk(val):
        color = RISK_COLORS.get(val, '#666')
        return f'background-color: {color}; color: white; font-weight: bold'

    styled = display_df.style.applymap(color_risk, subset=['Risk'])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Score Breakdown (for single wallet or clicked row) ───

    if mode == "Single Wallet":
        st.divider()
        st.subheader("Score Breakdown")

        components = result['components']
        cols = st.columns(len(components))
        for col, (name, value) in zip(cols, components.items()):
            weight = WEIGHTS[name]
            weighted = value * weight
            col.metric(
                name.replace('_', ' ').title(),
                f"{weighted:.2f}",
                delta=f"raw: {value:.2f} x {weight}",
            )

        # Wallet details
        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Win Rate Stats")
            wr = result['win_rate_stats']
            st.write(f"**Record:** {wr['wins']}W / {wr['losses']}L "
                     f"({wr['total']} markets)")
            st.write(f"**Win Rate:** {wr['win_rate']:.1%}")
            st.write(f"**p-value:** {wr['pvalue']:.6f}")
            st.write(f"**Realized PnL:** ${wr['realized_pnl']:,.2f}")
            st.write(f"**Avg PnL/Market:** ${wr['avg_pnl_per_market']:,.2f}")

        with col_b:
            st.subheader("Wallet Profile")
            age = result['age_info']
            bet = result['bet_stats']
            conc = result['concentration']
            st.write(f"**Wallet Age:** {age.get('age_days', '?')} days")
            st.write(f"**First Activity:** {age.get('first_activity', 'unknown')}")
            st.write(f"**Avg Bet Size:** ${bet['avg_bet_size']:,.0f}")
            st.write(f"**Total Volume:** ${bet['total_volume']:,.0f}")
            st.write(f"**Total Trades:** {bet['num_trades']}")
            st.write(f"**Top-3 PnL Share:** {conc['top3_pnl_share']:.0%}")
            st.write(f"**Markets Traded:** {conc['num_markets']}")

    # ── Download ─────────────────────────────────────────────

    st.divider()
    csv = df.to_csv(index=False)
    st.download_button(
        "Download CSV",
        csv,
        "polymarket_insider_scan.csv",
        "text/csv",
        use_container_width=True,
    )

else:
    # Landing state
    st.info("Configure scan settings in the sidebar and click **Run Scan** to begin.")

    st.markdown("""
### How it works

This tool screens Polymarket wallets for patterns consistent with insider trading:

1. **Leaderboard Screening** — Pulls top PnL wallets from politics, economics,
   and finance categories
2. **Statistical Win Rate** — Uses a binomial test to determine if a wallet's
   win rate is statistically improbable (not just lucky)
3. **Wallet Age** — Fresh accounts (<30 days) with outsized returns are flagged
4. **Pre-Resolution Timing** — Detects trades placed within 24h of market resolution
5. **PnL Concentration** — Wallets with PnL concentrated in few markets suggest
   targeted insider bets
6. **Category Focus** — Heavy focus on politics/economics/finance (vs. sports/crypto)

Each signal is scored 0-1 and weighted into a composite suspicion score (0-10).

### Data Source

All data comes from Polymarket's public APIs — no authentication or API keys required.
The [Data API](https://docs.polymarket.com) provides trade history, positions, and
leaderboard data. The [Gamma API](https://gamma-api.polymarket.com) provides market metadata.
""")
