"""
alt_data.py
Non-price data from free public sources, cached as CSV under data/alt/.

Crypto
- fear_greed():          Crypto Fear & Greed index (alternative.me), daily, Feb 2018+.
- deribit_funding(coin): BTC / ETH perpetual funding (Deribit), hourly -> daily mean, 2019+.
- hyperliquid_funding(coin): per-coin perpetual funding (Hyperliquid), hourly -> daily mean, May 2023+.
Stocks
- earnings(symbol):      report dates with EPS estimate / reported / surprise % (yfinance; needs lxml).
- analyst_actions(symbol): rating upgrades / downgrades and price-target changes (yfinance), 2012+.
- fred(series):          macro series from FRED (no API key needed).

Binance and Bybit would be better funding sources but block some regions; Hyperliquid and
Deribit are used instead. Every loader returns raw data indexed by the time it became known.
Turning that into features with no look-ahead is done in features_alt().
"""

from __future__ import annotations

import io
import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

ALT_DIR = os.path.join("data", "alt")
FRESH_HOURS = 24
HEADERS = {"User-Agent": "crypto-stock-ml-research/1.0"}


# ==========================================================
# CACHE
# ==========================================================
def _cached(name: str, fetch, fresh_hours: float = FRESH_HOURS) -> pd.DataFrame:
    """Return cached CSV if fresh, else fetch and cache. Falls back to a stale cache on failure."""
    os.makedirs(ALT_DIR, exist_ok=True)
    path = os.path.join(ALT_DIR, f"{name}.csv")
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) / 3600 < fresh_hours:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    try:
        df = fetch()
        if df is None or df.empty:
            raise ValueError("empty response")
        df.to_csv(path)
        return df
    except Exception as e:
        if os.path.exists(path):
            print(f"   [WARN] {name}: download failed ({e}); using stale cache")
            return pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"   [WARN] {name}: download failed ({e}); no data")
        return pd.DataFrame()


def _get(url: str, **kw):
    r = requests.get(url, headers=HEADERS, timeout=30, **kw)
    r.raise_for_status()
    return r


# ==========================================================
# CRYPTO
# ==========================================================
def fear_greed() -> pd.DataFrame:
    def fetch():
        d = _get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"}).json()["data"]
        df = pd.DataFrame(d)
        df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        return df.set_index("date")[["value"]].astype(float).rename(columns={"value": "fng"}).sort_index()
    return _cached("fear_greed", fetch)


def deribit_funding(coin: str = "BTC") -> pd.DataFrame:
    """Daily mean of Deribit's 8-hour-equivalent funding rate for BTC or ETH perpetuals."""
    def fetch():
        url = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
        start = pd.Timestamp("2019-01-01")
        end = pd.Timestamp.utcnow().tz_localize(None)
        rows = []
        t = start
        while t < end:
            t2 = min(t + pd.Timedelta(days=30), end)
            res = _get(url, params={"instrument_name": f"{coin}-PERPETUAL",
                                    "start_timestamp": int(t.timestamp() * 1000),
                                    "end_timestamp": int(t2.timestamp() * 1000)}).json().get("result", [])
            rows += [(r["timestamp"], r["interest_8h"]) for r in res]
            t = t2
            time.sleep(0.1)
        df = pd.DataFrame(rows, columns=["ts", "funding"]).drop_duplicates("ts")
        df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.floor("D")
        return df.groupby("date")[["funding"]].mean()
    return _cached(f"deribit_funding_{coin}", fetch)


def hyperliquid_funding(coin: str) -> pd.DataFrame:
    """Daily mean of Hyperliquid's hourly funding rate (scaled x8 to match the usual 8-hour quote)."""
    def fetch():
        url = "https://api.hyperliquid.xyz/info"
        start = int(pd.Timestamp("2023-01-01").timestamp() * 1000)
        now = int(time.time() * 1000)
        rows = []
        while start < now:
            for attempt in range(6):  # public API rate limit: back off and retry
                r = requests.post(url, json={"type": "fundingHistory", "coin": coin, "startTime": start},
                                  headers=HEADERS, timeout=30)
                if r.status_code != 429:
                    break
                time.sleep(2 ** attempt * 2)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows += [(b["time"], float(b["fundingRate"])) for b in batch]
            last = batch[-1]["time"]
            if last <= start:
                break
            start = last + 1
            time.sleep(0.5)
        df = pd.DataFrame(rows, columns=["ts", "funding"]).drop_duplicates("ts")
        df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.floor("D")
        return df.groupby("date")[["funding"]].mean() * 8
    return _cached(f"hyperliquid_funding_{coin}", fetch)


# ==========================================================
# STOCKS
# ==========================================================
def earnings(symbol: str) -> pd.DataFrame:
    """Past earnings reports. Index = report timestamp converted to UTC (naive)."""
    def fetch():
        import yfinance as yf
        e = yf.Ticker(symbol).get_earnings_dates(limit=100)
        if e is None or e.empty:
            return pd.DataFrame()
        e = e.dropna(subset=["Reported EPS"]).copy()
        e.index = pd.DatetimeIndex(e.index).tz_convert("UTC").tz_localize(None)
        e.index.name = "date"
        return e.rename(columns={"EPS Estimate": "eps_est", "Reported EPS": "eps",
                                 "Surprise(%)": "surprise_pct"}).sort_index()
    return _cached(f"earnings_{symbol}", fetch, fresh_hours=24 * 7)


def analyst_actions(symbol: str) -> pd.DataFrame:
    def fetch():
        import yfinance as yf
        u = yf.Ticker(symbol).upgrades_downgrades
        if u is None or u.empty:
            return pd.DataFrame()
        u = u.copy()
        u.index = pd.DatetimeIndex(u.index)
        if u.index.tz is not None:
            u.index = u.index.tz_convert("UTC").tz_localize(None)
        u.index.name = "date"
        return u[["Action", "priceTargetAction"]].sort_index()
    return _cached(f"analyst_{symbol}", fetch, fresh_hours=24 * 7)


def fred(series: str) -> pd.DataFrame:
    def fetch():
        txt = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series}).text
        df = pd.read_csv(io.StringIO(txt))
        df.columns = ["date", series]
        df["date"] = pd.to_datetime(df["date"])
        df[series] = pd.to_numeric(df[series], errors="coerce")
        return df.set_index("date").dropna()
    return _cached(f"fred_{series}", fetch)


# ==========================================================
# FEATURES  (all strictly known before the decision bar's close)
# ==========================================================
def _asof(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Last value on or before each date."""
    s = series.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(dates)).ffill().reindex(dates)


def crypto_market_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Market-wide crypto sentiment, same value for every coin on a date."""
    out = pd.DataFrame(index=dates)
    fg = fear_greed()
    if not fg.empty:
        f = fg["fng"]
        out["fng"] = _asof(f, dates)
        out["fng_chg7"] = _asof(f - f.shift(7), dates)
    btc = deribit_funding("BTC")
    if not btc.empty:
        # day d's funding is fully known at the end of day d (= the crypto daily close)
        f = btc["funding"]
        out["btc_fund7"] = _asof(f.rolling(7).mean(), dates)
        out["btc_fund_z90"] = _asof((f - f.rolling(90).mean()) / f.rolling(90).std(), dates)
    return out


def coin_funding_features(coin: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=dates)
    hl = hyperliquid_funding(coin)
    if hl.empty:
        return out
    f = hl["funding"]
    out["fund1"] = _asof(f, dates)
    out["fund7"] = _asof(f.rolling(7).mean(), dates)
    out["fund_z90"] = _asof((f - f.rolling(90, min_periods=30).mean()) / f.rolling(90, min_periods=30).std(), dates)
    first = f.index.min()
    out.loc[out.index < first + pd.Timedelta(days=7)] = np.nan
    return out


def stock_event_features(symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Earnings and analyst features. Events are only used from the calendar day AFTER they
    happened (a report at 16:00 on day d is not known at day d's close), which is slightly
    conservative for pre-market reports.
    """
    out = pd.DataFrame(index=dates)
    e = earnings(symbol)
    if not e.empty:
        known = e.copy()
        known.index = known.index.normalize() + pd.Timedelta(days=1)
        last_date = pd.Series(known.index, index=known.index)
        out["eps_surprise"] = _asof(known["surprise_pct"].clip(-100, 100), dates)
        since = pd.Series(dates, index=dates) - _asof(last_date, dates)
        out["days_since_earn"] = since.dt.days.clip(upper=120)
        prev = known["surprise_pct"].shift(1)
        out["eps_surprise_prev"] = _asof(prev.clip(-100, 100), dates)
    a = analyst_actions(symbol)
    if not a.empty:
        a = a.copy()
        a.index = a.index.normalize() + pd.Timedelta(days=1)
        score = (a["Action"].map({"up": 1, "down": -1}).fillna(0)
                 + a["priceTargetAction"].map({"Raises": 0.5, "Lowers": -0.5}).fillna(0))
        daily = score.groupby(level=0).sum()
        full = daily.reindex(pd.date_range(min(daily.index.min(), dates.min()), dates.max(), freq="D"), fill_value=0)
        out["analyst_net30"] = full.rolling(30).sum().reindex(dates)
        out["analyst_net90"] = full.rolling(90).sum().reindex(dates)
        out.loc[out.index < daily.index.min() + pd.Timedelta(days=90), ["analyst_net30", "analyst_net90"]] = np.nan
    return out


def stock_macro_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """FRED daily series, shifted one day (published after the US close)."""
    out = pd.DataFrame(index=dates)
    for s, name in [("T10Y2Y", "curve_10y2y"), ("BAA10Y", "baa_spread")]:
        df = fred(s)
        if df.empty:
            continue
        v = df[s].copy()
        v.index = v.index + pd.Timedelta(days=1)
        out[name] = _asof(v, dates)
        out[name + "_chg20"] = _asof(v - v.shift(20), dates)
    return out
