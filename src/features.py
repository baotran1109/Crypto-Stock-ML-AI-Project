"""
features.py
Compute indicators and prepare supervised learning dataset (pure pandas).
"""

import numpy as np
import pandas as pd

# Expanded feature list
FEATURE_COLS = [
    "rsi", "ema12", "ema26", "macd", "macd_signal", "macd_hist",
    "bb_low", "bb_mid", "bb_high",
    "ret1", "ret5", "vol20", "ema_ratio", "bb_width", "above_ema"
]

def _as_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(index=df.index, dtype="float64")
    col = df[name]
    if isinstance(col, pd.DataFrame):
        for c in reversed(col.columns.tolist()):
            s = pd.to_numeric(col[c], errors="coerce")
            if s.notna().any():
                return s
        return pd.to_numeric(col.iloc[:, 0], errors="coerce")
    return pd.to_numeric(col, errors="coerce")

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def _rsi(s: pd.Series, length: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _bbands(s: pd.Series, length: int = 20, n_std: float = 2.0):
    ma = s.rolling(length, min_periods=length).mean()
    sd = s.rolling(length, min_periods=length).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    return lower, ma, upper

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="last")]

    close = _as_series(df, "close")
    open_ = _as_series(df, "open")
    high = _as_series(df, "high")
    low = _as_series(df, "low")
    volume = _as_series(df, "volume")

    core = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }).dropna(subset=["close"])

    # Require at least 60 rows to compute indicators safely
    if core["close"].notna().sum() < 60:
        return pd.DataFrame(index=core.index)

    # Core indicators
    core["ema12"] = _ema(core["close"], 12)
    core["ema26"] = _ema(core["close"], 26)
    core["rsi"] = _rsi(core["close"], 14)

    macd_line = core["ema12"] - core["ema26"]
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - macd_signal
    core["macd"] = macd_line
    core["macd_signal"] = macd_signal
    core["macd_hist"] = macd_hist

    bb_low, bb_mid, bb_high = _bbands(core["close"], length=20, n_std=2.0)
    core["bb_low"] = bb_low
    core["bb_mid"] = bb_mid
    core["bb_high"] = bb_high

    # === New engineered features ===
    core["ret1"] = core["close"].pct_change(1)
    core["ret5"] = core["close"].pct_change(5)
    core["vol20"] = core["ret1"].rolling(20).std()

    core["ema_ratio"] = core["ema12"] / core["ema26"] - 1
    core["bb_width"] = (core["bb_high"] - core["bb_low"]) / core["bb_mid"]
    core["above_ema"] = (core["close"] > core["ema26"]).astype(int)

    # Drop only rows where the core features are all NaN
    core = core.dropna(thresh=int(len(core.columns) * 0.5))

    return core

def make_supervised(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    X = df[FEATURE_COLS].shift(1).dropna()
    # Predict 5-day ahead movement
    y = (df["close"].shift(-5) > df["close"]).astype(int).loc[X.index]
    return X, y
