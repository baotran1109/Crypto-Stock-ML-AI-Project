"""
data_loader.py
Unified OHLCV loader with Yahoo -> CoinGecko fallback, caching, retries, and normalization.

Features:
--------
- Public entrypoint: load_crypto(symbol, lookback="5y", prefer="auto", use_cache=True, fresh_hours=12, force_refresh=False)
- Sources:
    * Yahoo Finance (yfinance) for OHLCV (works for stocks & many cryptos)
    * CoinGecko OHLC (no reliable volume; we return NaN for volume)
- Robustness:
    * CSV cache under ./data with freshness window (hours)
    * HTTP session with retry/backoff & custom User-Agent for CoinGecko
    * Graceful fallback & clear exceptions
- Normalization:
    * Columns: open, high, low, close, volume, source, symbol
    * Index: UTC -> tz-naive (so you can merge/join safely)
- CLI smoke test: `python -m src.data_loader`
"""

from __future__ import annotations

import errno
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import pandas as pd

# ------------------------ Config ------------------------
CACHE_DIR = "data"
YF_INTERVAL = "1d"
YF_PERIOD_DEFAULT = "5y"
CG_DAYS_DEFAULT = "max"

USER_AGENT = "ml-edu-project/1.0 (+https://example.edu)"

SYMBOL_TO_CGID = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "MATIC": "matic-network",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "LTC": "litecoin",
    "TRX": "tron",
    "BNB": "binancecoin",
    "AVAX": "avalanche-2",
    "SHIB": "shiba-inu",
}

# ------------------------ Errors ------------------------

class DataLoadError(RuntimeError):
    """Raised when both Yahoo and CoinGecko fail (or are disallowed)."""

class UnknownSymbolError(ValueError):
    """Raised when we can't map a ticker to a CoinGecko ID."""

# ------------------------ Utils -------------------------
def _ensure_dir(path: str) -> None:
    try: 
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _cache_path(symbol: str, source: str, lookback: str) -> str:
    safe = symbol.upper().replace("/", "_").replace("-", "_")
    return os.path.join(CACHE_DIR, f"{safe}__{source}__{lookback}.csv")

def _is_fresh(path: str, max_age_hours: int) -> bool:
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.now() - mtime) < timedelta(hours=max_age_hours)

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure lowercase canonical names and a consistent column set.
    Returns only: open, high, low, close, volume
    """
    df = df.rename(columns=str.lower)
    cols = ["open", "high", "low", "close", "volume"]
    for c in cols:
        if c not in df.columns:
            df[c] = float("nan")
    return df[cols]

def _normalize_index_utc_tznaive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make index UTC tz-naive for safe merges across sources.
    - If tz-aware: convert to UTC then drop tz
    - If tz-naive: leave as-is (assume UTC-like)
    """
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            df.index = idx.tz_convert(timezone.utc).tz_localize(None)
    return df
    
def _map_symbol_to_cgid(symbol: str) -> str:
    s = symbol.upper().replace("-USD", "").replace("/USD", "")
    if s in SYMBOL_TO_CGID:
        return SYMBOL_TO_CGID[s]
    raise UnknownSymbolError(
        f"Unknown symbol '{symbol}'. Add mapping in SYMBOL_TO_CGID (e.g., '{s}': 'coingecko-coin-id')."
    )

def _lookback_to_cg_days(lookback: str) -> str:
    """
     Convert Yahoo-esque lookbacks ('1y','2y','max') to CoinGecko 'days' strings.
    '5y+' -> 'max'; '1y' -> '365'; '2y' -> '730'; default -> '365'
    """ 
    lookback = lookback.strip().lower()
    if lookback == "max":
        return "max"
    
    if lookback.endswith("y") and lookback[:-1].isdigit():
        years = int(lookback[:-1])
        return "max" if years >= 5 else str(years * 365)
    
    if lookback.endswith("d") and lookback[:-1].isdigit():
        return lookback[:-1]
    
    return "365"

# ------------------------ Yahoo Finance ------------------------
def _fetch_yahoo(symbol: str, period: str = YF_PERIOD_DEFAULT, interval: str = YF_INTERVAL) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol, period=period, interval=interval, progress=False,auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna()
    df.index.name = "time"
    df = _normalize_columns(df)
    df = _normalize_index_utc_tznaive(df)
    return df


# ------------------------ CoinGecko -----------------------------
def _request_session_with_retries():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )

    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": USER_AGENT})
    return s

def _fetch_coingecko(symbol: str, days: str = CG_DAYS_DEFAULT, timeout: int = 30) -> pd.DataFrame:
    import requests

    coin_id = _map_symbol_to_cgid(symbol)
    session = _request_session_with_retries()

    ohlc_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    ohlc_params = {"vs_currency": "usd", "days": days}

    try:
        r = session.get(ohlc_url, params=ohlc_params, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                df = pd.DataFrame(data, columns=["time","open","high","low","close"])
                df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
                df = df.set_index("time").sort_index()
                df["volume"] = float("nan")
                df = _normalize_columns(df)
                df = _normalize_index_utc_tznaive(df)

                df.attrs["synthetic_ohlc"] = False
                return df
    except requests.RequestException:
        pass 


    mc_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    mc_params = {"vs_currency": "usd", "days": days}
    try:
        r = session.get(mc_url, params=mc_params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        prices = data.get("prices", [])
        if not prices:
            return pd.DataFrame()
        df = pd.DataFrame(prices, columns=["time","close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df = df.set_index("time").sort_index()

        out = pd.DataFrame(index=df.index)
        out["open"] = df["close"]
        out["high"] = df["close"]
        out["low"] = df["close"]
        out["close"] = df["close"]
        out["volume"] = float("nan")

        out = _normalize_columns(out)
        out = _normalize_index_utc_tznaive(out)

        out.attrs["synthetic_ohlc"] = True
        return out
    except requests.RequestException:
        return pd.DataFrame()


# ------------------------ Public API ----------------------------
@dataclass
class LoadResult:
    df: pd.DataFrame
    source: Literal["yahoo", "coingecko"]

def load_crypto(
        symbol: str,
        lookback: str = YF_PERIOD_DEFAULT,
        prefer: Literal["auto", "yahoo", "coingecko"] = "auto",
        use_cache: bool = True,
        fresh_hours: int = 12,
        force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load OHLCV for a crypto (or stock via Yahoo) with fallback.
    Returns a DataFrame indexed by time with columns:
      open, high, low, close, volume, source, symbol

    Parameters
    ----------
    symbol : str
        e.g., 'BTC-USD', 'ETH-USD', 'AAPL'
    lookback : str
        Yahoo period ('1y','5y','10y','max'); for CoinGecko it's mapped to 'days'
    prefer : {'yahoo','coingecko','auto'}
        - 'auto': try Yahoo first, then CoinGecko
        - 'yahoo': use only Yahoo
        - 'coingecko': use only CoinGecko
    use_cache : bool
        Read/write CSV cache under ./data
    fresh_hours : int
        Cache freshness window (hours)
    force_refresh : bool
        Ignore cache and pull fresh from source(s)
    """
    _ensure_dir(CACHE_DIR)

    if prefer in ("auto", "yahoo"):
        cache_y = _cache_path(symbol, "yahoo", lookback)
        if use_cache and not force_refresh and _is_fresh(cache_y, fresh_hours):
            df_y = pd.read_csv(cache_y, parse_dates=["time"]).set_index("time")
            df_y = _normalize_columns(df_y)
            df_y = _normalize_index_utc_tznaive(df_y)
            df_y["source"] = "yahoo"
            df_y["symbol"] = symbol.upper()
            return df_y
        
        try:
            df_y = _fetch_yahoo(symbol, period=lookback)
            if not df_y.empty:
                df_y["source"] = "yahoo"
                df_y["symbol"] = symbol.upper()
                if use_cache:
                    df_y.reset_index().to_csv(cache_y, index=False)
                return df_y
        except Exception:
            if prefer == "yahoo":
                raise
    
    if prefer in ("auto", "coingecko"):
        cg_days = _lookback_to_cg_days(lookback)
        cache_cg = _cache_path(symbol, "coingecko", str(cg_days))
        if use_cache and not force_refresh and _is_fresh(cache_cg, fresh_hours):
            df_cg = pd.read_csv(cache_cg, parse_dates=["time"]).set_index("time")
            df_cg = _normalize_columns(df_cg)
            df_cg = _normalize_index_utc_tznaive(df_cg)
            df_cg["source"] = "coingecko"
            df_cg["symbol"] = symbol.upper()
            return df_cg
        
        try:
            df_cg = _fetch_coingecko(symbol, days=cg_days)
            if not df_cg.empty:
                df_cg["source"] = "coingecko"
                df_cg["symbol"] = symbol.upper()
                if use_cache:
                    df_cg.reset_index().to_csv(cache_cg, index=False)
                return df_cg
        except UnknownSymbolError:
            if prefer == "coingecko":
                raise

        except Exception:
            if prefer == "coingecko":
                raise
        
    raise DataLoadError(f"Failed to load data for symbol '{symbol}' from preferred source(s).")
# ------------------------ CLI ----------------------------

def _smoke():
    tests = ("BTC-USD", "ETH-USD", "SHIB-USD", "AAPL")
    for sym in tests:
        try:
            df=load_crypto(sym, lookback="5y", prefer="auto", use_cache=True, fresh_hours=12)
            src= df["source"].iloc[-1]
            print(f"{sym:8s} -> rows: {len(df):5d} | source={src}")
            print(df.tail(2))
        except Exception as e:
            print(f"{sym:8s} -> ERROR: {e}")

if __name__ == "__main__":
    _ensure_dir(CACHE_DIR)
    _smoke()

