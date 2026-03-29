"""
Polymarket Insider Wallet Detector — Core Engine
=================================================
Screens Polymarket wallets for insider trading signals by combining:
  1. Leaderboard screening — high-PnL wallets across target categories
  2. Win rate analysis — binomial p-value to test statistical significance
  3. Wallet age — flag fresh accounts with outsized returns
  4. Timing analysis — trades placed suspiciously close to resolution
  5. Concentration — few markets, large sizes, niche bets
  6. Composite scoring — weighted suspicion score (0-10)

All data comes from Polymarket's public APIs (no auth needed).
"""

import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================

DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"

REQUEST_DELAY = 0.35
REQUEST_TIMEOUT = 20

TARGET_CATEGORIES = ["POLITICS", "ECONOMICS", "FINANCE"]

# Detection thresholds
MIN_PNL_USD = 500
MIN_MARKETS = 3
MAX_WALLET_AGE_DAYS = 180
MIN_AVG_BET_SIZE = 200
PRE_RESOLUTION_HOURS = 24
CONCENTRATION_THRESH = 0.6

# Scoring weights (sum to 10)
WEIGHTS = {
    'win_rate':       2.5,
    'wallet_age':     2.0,
    'timing':         2.0,
    'concentration':  1.5,
    'bet_size':       1.0,
    'category_focus': 1.0,
}

# ============================================================
# API CLIENT
# ============================================================

class PolymarketClient:
    """Lightweight client for Polymarket public APIs (no auth needed)."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "polymarket-insider-detector/1.0",
        })
        self._market_cache: Dict[str, dict] = {}

    def _get(self, url: str, params: dict = None):
        time.sleep(REQUEST_DELAY)
        resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def get_leaderboard(self, category: str = None, limit: int = 50,
                        offset: int = 0, order_by: str = "pnl") -> list:
        params = {"limit": limit, "offset": offset, "orderBy": order_by}
        if category:
            params["category"] = category
        return self._get(f"{DATA_API}/v1/leaderboard", params)

    def get_trades(self, address: str, limit: int = 100, offset: int = 0) -> list:
        return self._get(f"{DATA_API}/trades",
                         {"user": address, "limit": limit, "offset": offset})

    def get_all_trades(self, address: str, max_pages: int = 20) -> list:
        all_trades = []
        for page in range(max_pages):
            batch = self.get_trades(address, limit=100, offset=page * 100)
            if not batch:
                break
            all_trades.extend(batch)
            if len(batch) < 100:
                break
        return all_trades

    def get_closed_positions(self, address: str, limit: int = 50,
                             offset: int = 0) -> list:
        # IMPORTANT: must use sortBy=timestamp to get wins AND losses.
        # Default sort is by realizedPnl descending, which returns only
        # winners first — paginating never reaches the losses for large wallets.
        return self._get(f"{DATA_API}/closed-positions",
                         {"user": address, "limit": limit, "offset": offset,
                          "sortBy": "timestamp"})

    def get_all_closed_positions(self, address: str, max_pages: int = 100) -> list:
        all_pos = []
        page_size = 50
        for page in range(max_pages):
            batch = self.get_closed_positions(address, limit=page_size,
                                              offset=page * page_size)
            if not batch:
                break
            all_pos.extend(batch)
            if len(batch) < page_size:
                break
        return all_pos

    def get_first_activity(self, address: str) -> Optional[dict]:
        result = self._get(f"{DATA_API}/activity",
                           {"user": address, "limit": 1, "sortDirection": "ASC"})
        return result[0] if result else None

    def get_market(self, condition_id: str) -> dict:
        if condition_id in self._market_cache:
            return self._market_cache[condition_id]
        try:
            markets = self._get(f"{GAMMA_API}/markets",
                                {"condition_id": condition_id})
            if markets:
                market = markets[0] if isinstance(markets, list) else markets
                self._market_cache[condition_id] = market
                return market
        except Exception:
            pass
        return {}


# ============================================================
# ANALYSIS ENGINE
# ============================================================

def parse_timestamp(ts) -> Optional[datetime]:
    if not ts:
        return None
    if isinstance(ts, (int, float)):
        if ts > 1e12:
            ts = ts / 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, str):
        for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"]:
            try:
                dt = datetime.strptime(ts, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        try:
            val = float(ts)
            if val > 1e12:
                val = val / 1000
            return datetime.fromtimestamp(val, tz=timezone.utc)
        except ValueError:
            pass
    return None


def compute_win_rate_stats(closed_positions: list) -> dict:
    if not closed_positions:
        return {'wins': 0, 'losses': 0, 'total': 0, 'win_rate': 0,
                'pvalue': 1.0, 'realized_pnl': 0, 'avg_pnl_per_market': 0}

    wins = losses = 0
    total_pnl = 0.0

    for pos in closed_positions:
        pnl = float(pos.get('realizedPnl', 0) or 0)
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

    total = wins + losses
    if total == 0:
        return {'wins': 0, 'losses': 0, 'total': 0, 'win_rate': 0,
                'pvalue': 1.0, 'realized_pnl': total_pnl, 'avg_pnl_per_market': 0}

    win_rate = wins / total
    pvalue = stats.binomtest(wins, total, 0.5, alternative='greater').pvalue

    return {
        'wins': wins, 'losses': losses, 'total': total,
        'win_rate': win_rate, 'pvalue': pvalue,
        'realized_pnl': total_pnl, 'avg_pnl_per_market': total_pnl / total,
    }


def compute_wallet_age(client: PolymarketClient, address: str) -> dict:
    first = client.get_first_activity(address)
    if not first:
        return {'first_activity': None, 'age_days': None}

    ts = parse_timestamp(first.get('timestamp') or first.get('createdAt'))
    if not ts:
        return {'first_activity': None, 'age_days': None}

    age = datetime.now(tz=timezone.utc) - ts
    return {'first_activity': ts, 'age_days': age.days}


def compute_timing_score(trades: list, closed_positions: list,
                         client: PolymarketClient = None) -> dict:
    pre_res_count = 0
    hours_before = []
    total_checked = 0

    for pos in closed_positions:
        end_str = pos.get('endDate')
        trade_ts_raw = pos.get('timestamp')
        if not end_str or not trade_ts_raw:
            continue

        end_dt = parse_timestamp(end_str)
        trade_ts = parse_timestamp(trade_ts_raw)
        if not end_dt or not trade_ts:
            continue

        total_checked += 1
        delta = end_dt - trade_ts
        hours = delta.total_seconds() / 3600

        if 0 < hours <= PRE_RESOLUTION_HOURS:
            pre_res_count += 1
            hours_before.append(hours)

    if total_checked < 5 and trades and client:
        market_end_cache = {}
        for trade in trades[:30]:
            cid = trade.get('conditionId') or trade.get('condition_id')
            if not cid:
                continue
            trade_ts = parse_timestamp(trade.get('timestamp'))
            if not trade_ts:
                continue
            if cid not in market_end_cache:
                market = client.get_market(cid)
                market_end_cache[cid] = parse_timestamp(
                    market.get('endDate') or market.get('end_date_iso'))
            end_dt = market_end_cache.get(cid)
            if not end_dt:
                continue
            total_checked += 1
            delta = end_dt - trade_ts
            hours = delta.total_seconds() / 3600
            if 0 < hours <= PRE_RESOLUTION_HOURS:
                pre_res_count += 1
                hours_before.append(hours)

    pre_res_pct = pre_res_count / total_checked if total_checked > 0 else 0
    avg_hours = np.mean(hours_before) if hours_before else None

    return {
        'pre_resolution_trades': pre_res_count,
        'total_trades': total_checked,
        'pre_resolution_pct': pre_res_pct,
        'avg_hours_before_resolution': avg_hours,
    }


def compute_concentration(closed_positions: list) -> dict:
    if not closed_positions:
        return {'top3_pnl_share': 0, 'num_markets': 0, 'hhi': 0}

    pnls = [abs(float(pos.get('realizedPnl', 0) or 0))
            for pos in closed_positions]
    pnls = [p for p in pnls if p > 0]

    if not pnls:
        return {'top3_pnl_share': 0, 'num_markets': 0, 'hhi': 0}

    pnls.sort(reverse=True)
    total = sum(pnls)
    top3_share = sum(pnls[:3]) / total if total > 0 else 0
    shares = [p / total for p in pnls]
    hhi = sum(s ** 2 for s in shares)

    return {'top3_pnl_share': top3_share, 'num_markets': len(pnls), 'hhi': hhi}


def compute_bet_size_stats(trades: list) -> dict:
    if not trades:
        return {'avg_bet_size': 0, 'max_bet_size': 0, 'total_volume': 0,
                'num_trades': 0}

    sizes = []
    for t in trades:
        size = float(t.get('size', 0) or 0)
        price = float(t.get('price', 1) or 1)
        usd_size = size * price
        if usd_size > 0:
            sizes.append(usd_size)

    if not sizes:
        return {'avg_bet_size': 0, 'max_bet_size': 0, 'total_volume': 0,
                'num_trades': 0}

    return {
        'avg_bet_size': np.mean(sizes), 'max_bet_size': max(sizes),
        'total_volume': sum(sizes), 'num_trades': len(sizes),
    }


def compute_category_focus(trades: list, client: PolymarketClient) -> dict:
    if not trades:
        return {'insider_category_pct': 0, 'categories': {}}

    insider_tags = {'politics', 'economics', 'finance', 'macro', 'rates',
                    'war', 'military', 'oil', 'gas', 'energy', 'federal reserve',
                    'fed', 'inflation', 'gdp', 'employment', 'tariff', 'trade',
                    'government', 'election', 'geopolitics'}

    cat_counts = {}
    insider_count = 0
    total_checked = 0
    seen_conditions = set()

    for trade in trades:
        condition_id = trade.get('conditionId') or trade.get('condition_id')
        if not condition_id or condition_id in seen_conditions:
            continue
        seen_conditions.add(condition_id)

        market = client.get_market(condition_id)
        if not market:
            continue

        total_checked += 1
        tags = [t.lower() for t in (market.get('tags', []) or [])]
        question = (market.get('question', '') or '').lower()
        group_title = (market.get('groupItemTitle', '') or '').lower()
        category = (market.get('category', '') or '').lower()
        all_text = ' '.join(tags + [question, group_title, category])

        for tag in tags:
            cat_counts[tag] = cat_counts.get(tag, 0) + 1

        if any(kw in all_text for kw in insider_tags):
            insider_count += 1

    insider_pct = insider_count / total_checked if total_checked > 0 else 0

    return {
        'insider_category_pct': insider_pct,
        'categories': dict(sorted(cat_counts.items(), key=lambda x: -x[1])[:10]),
    }


def compute_suspicion_score(wr_stats: dict, age_info: dict, timing: dict,
                            concentration: dict, bet_stats: dict,
                            category_info: dict) -> Tuple[float, dict]:
    components = {}

    # Win rate
    if wr_stats['total'] >= MIN_MARKETS and wr_stats['pvalue'] < 0.5:
        wr_score = min(1.0, -np.log10(max(wr_stats['pvalue'], 1e-10)) / 4)
    elif wr_stats['win_rate'] >= 0.9 and wr_stats['total'] >= MIN_MARKETS:
        wr_score = 0.8
    else:
        wr_score = 0
    components['win_rate'] = wr_score

    # Wallet age
    if age_info.get('age_days') is not None:
        age = age_info['age_days']
        if age <= 7:
            age_score = 1.0
        elif age <= 30:
            age_score = 0.8
        elif age <= 90:
            age_score = 0.5
        elif age <= MAX_WALLET_AGE_DAYS:
            age_score = 0.2
        else:
            age_score = 0
    else:
        age_score = 0.3
    components['wallet_age'] = age_score

    # Timing
    pre_res_pct = timing.get('pre_resolution_pct', 0)
    if pre_res_pct >= 0.8:
        timing_score = 1.0
    elif pre_res_pct >= 0.5:
        timing_score = 0.7
    elif pre_res_pct >= 0.3:
        timing_score = 0.4
    else:
        timing_score = pre_res_pct
    components['timing'] = timing_score

    # Concentration
    top3 = concentration.get('top3_pnl_share', 0)
    num_mkts = concentration.get('num_markets', 0)
    if top3 >= CONCENTRATION_THRESH and num_mkts <= 10:
        conc_score = min(1.0, top3)
    elif top3 >= 0.4:
        conc_score = top3 * 0.7
    else:
        conc_score = 0
    components['concentration'] = conc_score

    # Bet size
    avg_bet = bet_stats.get('avg_bet_size', 0)
    if avg_bet >= 5000:
        bet_score = 1.0
    elif avg_bet >= 1000:
        bet_score = 0.7
    elif avg_bet >= MIN_AVG_BET_SIZE:
        bet_score = 0.4
    else:
        bet_score = 0
    components['bet_size'] = bet_score

    # Category focus
    insider_pct = category_info.get('insider_category_pct', 0)
    if insider_pct >= 0.8:
        cat_score = 1.0
    elif insider_pct >= 0.5:
        cat_score = 0.7
    else:
        cat_score = insider_pct
    components['category_focus'] = cat_score

    total = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
    return total, components


def classify_risk(score: float) -> str:
    if score >= 7:
        return "CRITICAL"
    elif score >= 5:
        return "HIGH"
    elif score >= 3:
        return "MEDIUM"
    else:
        return "LOW"


# ============================================================
# SCANNER
# ============================================================

def scan_leaderboard(client: PolymarketClient, categories: list = None,
                     top_n: int = 50, min_pnl: float = MIN_PNL_USD,
                     progress_callback=None) -> list:
    categories = categories or TARGET_CATEGORIES
    seen = set()
    candidates = []

    for i, cat in enumerate(categories):
        if progress_callback:
            progress_callback(f"Scanning {cat} leaderboard...")
        try:
            lb = client.get_leaderboard(category=cat, limit=top_n, order_by="pnl")
        except Exception:
            continue

        if not lb:
            continue

        for entry in lb:
            addr = (entry.get('proxyWallet') or entry.get('userAddress')
                    or entry.get('address'))
            if not addr or addr in seen:
                continue
            seen.add(addr)

            pnl = float(entry.get('pnl', 0) or 0)

            if pnl >= min_pnl:
                candidates.append({
                    'address': addr,
                    'pnl': pnl,
                    'source_category': cat,
                    'username': entry.get('userName', '') or entry.get('username', ''),
                    'volume': float(entry.get('vol', 0) or entry.get('volume', 0) or 0),
                })

    return candidates


def analyze_wallet(client: PolymarketClient, address: str,
                   skip_timing: bool = False, skip_category: bool = False,
                   progress_callback=None) -> dict:
    result = {'address': address}

    if progress_callback:
        progress_callback(f"Fetching positions for {address[:10]}...")
    closed = client.get_all_closed_positions(address)
    wr_stats = compute_win_rate_stats(closed)
    result['win_rate_stats'] = wr_stats

    if progress_callback:
        progress_callback(f"Checking wallet age...")
    age_info = compute_wallet_age(client, address)
    result['age_info'] = age_info

    if progress_callback:
        progress_callback(f"Fetching trades...")
    trades = client.get_all_trades(address)
    bet_stats = compute_bet_size_stats(trades)
    result['bet_stats'] = bet_stats

    if not skip_timing:
        if progress_callback:
            progress_callback(f"Running timing analysis...")
        timing = compute_timing_score(trades, closed, client)
    else:
        timing = {'pre_resolution_trades': 0, 'total_trades': 0,
                  'pre_resolution_pct': 0, 'avg_hours_before_resolution': None}
    result['timing'] = timing

    concentration = compute_concentration(closed)
    result['concentration'] = concentration

    if not skip_category and trades:
        if progress_callback:
            progress_callback(f"Analyzing categories...")
        category_info = compute_category_focus(trades[:50], client)
    else:
        category_info = {'insider_category_pct': 0, 'categories': {}}
    result['category_info'] = category_info

    score, components = compute_suspicion_score(
        wr_stats, age_info, timing, concentration, bet_stats, category_info)
    result['score'] = score
    result['risk'] = classify_risk(score)
    result['components'] = components

    return result


def result_to_row(result: dict, username: str = '', source_category: str = '',
                  leaderboard_pnl: float = 0) -> dict:
    return {
        'address': result['address'],
        'username': username,
        'source_category': source_category,
        'score': result['score'],
        'risk': result['risk'],
        'win_rate': result['win_rate_stats']['win_rate'],
        'wins': result['win_rate_stats']['wins'],
        'losses': result['win_rate_stats']['losses'],
        'total_markets': result['win_rate_stats']['total'],
        'pvalue': result['win_rate_stats']['pvalue'],
        'realized_pnl': result['win_rate_stats']['realized_pnl'],
        'wallet_age_days': result['age_info'].get('age_days'),
        'avg_bet_size': result['bet_stats']['avg_bet_size'],
        'total_volume': result['bet_stats']['total_volume'],
        'num_trades': result['bet_stats']['num_trades'],
        'pre_res_pct': result['timing'].get('pre_resolution_pct', 0),
        'top3_pnl_share': result['concentration']['top3_pnl_share'],
        'num_markets_traded': result['concentration']['num_markets'],
        'insider_cat_pct': result['category_info']['insider_category_pct'],
        'leaderboard_pnl': leaderboard_pnl,
        'sc_win_rate': result['components']['win_rate'],
        'sc_wallet_age': result['components']['wallet_age'],
        'sc_timing': result['components']['timing'],
        'sc_concentration': result['components']['concentration'],
        'sc_bet_size': result['components']['bet_size'],
        'sc_category': result['components']['category_focus'],
    }
