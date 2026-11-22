"""
backtest.py
Backtest (with backtrader) a long-only strategy that trades on the ML model's predictions,
and compare it against simple rule-based benchmarks run through the exact same engine.

Key points
- Every strategy is expressed as a per-bar signal array: 1 = be long, 0 = be flat,
  -1 = no change. One backtrader engine runs all of them with the same capital,
  commission, fills and test window, so the comparison is apples to apples.
- Indicators are computed on the full history first, then the test window is sliced,
  so the window starts with valid indicators (no warm-up gap) and long moving
  averages (e.g. 200-day) are usable immediately.
- By default the backtest only covers dates AFTER the model's training period
  (out_of_sample=True), so results are not inflated by data the model has seen.
- Orders fill at the close of the signal bar (signals only use data up to that close),
  and position size is fractional, so expensive assets like BTC trade with small capital.
- Shuffle test: the ML signals are randomly permuted many times (same number of buy/sell/
  hold days, random timing). If the model beats most shuffles, its timing adds value.
"""

from typing import Dict, List, Optional

import backtrader as bt
import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_crypto, DataLoadError
from src.features import HORIZON, add_indicators

METADATA_PATH = "artifacts/training_metadata.pkl"
ML_NAME = "ML model"
N_SHUFFLES = 20
SHUFFLE_SEED = 42


def load_training_metadata(path: str = METADATA_PATH) -> Optional[dict]:
    try:
        return joblib.load(path)
    except (FileNotFoundError, OSError):
        return None


def _thresholds(confidence_threshold: float):
    """Map the 0-1 UI slider to buy/sell probability thresholds (asymmetric)."""
    delta = 0.01 + confidence_threshold * 0.09
    return 0.5 + delta, 0.5 - delta * 0.7


# ==========================================================
# SIGNALS  (1 = long, 0 = flat, -1 = keep current position)
# ==========================================================
def ml_signals(probs: np.ndarray, buy_threshold: float, sell_threshold: float) -> np.ndarray:
    sig = np.full(len(probs), -1)
    sig[probs > buy_threshold] = 1
    sig[probs < sell_threshold] = 0
    sig[np.isnan(probs)] = -1
    return sig


def benchmark_signals(full: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Rule-based benchmarks, computed on the FULL history (so long windows are warm),
    using only data up to each bar's close. Returned series are indexed like `full`.
    """
    close = full["close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    mom20 = close.pct_change(20)
    rsi = full["rsi"]

    def state(cond: pd.Series, defined: pd.Series) -> pd.Series:
        return pd.Series(np.where(defined, cond.astype(int), -1), index=full.index)

    rsi_sig = pd.Series(-1, index=full.index)
    rsi_sig[rsi < 30] = 1   # oversold -> buy
    rsi_sig[rsi > 55] = 0   # recovered -> exit

    return {
        "Buy & hold": pd.Series(1, index=full.index),
        "SMA 50/200 crossover": state(sma50 > sma200, sma200.notna()),
        "Momentum (20-day)": state(mom20 > 0, mom20.notna()),
        "RSI mean reversion": rsi_sig,
    }


BENCHMARK_RULES = {
    "Buy & hold": "Buy on the first day and hold to the end.",
    "SMA 50/200 crossover": "Long while the 50-day average is above the 200-day average.",
    "Momentum (20-day)": "Long while the price is above its level 20 days ago.",
    "RSI mean reversion": "Buy when RSI drops below 30, sell when it rises above 55.",
}


# ==========================================================
# ENGINE
# ==========================================================
class SignalStrategy(bt.Strategy):
    """Long-only: go long on signal 1, go flat on signal 0, otherwise hold."""

    params = (
        ("signals", None),      # np.ndarray aligned 1:1 with the data feed
        ("confidence", None),   # optional np.ndarray (ML probabilities) stored on trades
        ("closes", None),       # np.ndarray of closes, for the horizon-ahead check
        ("commission", 0.001),
    )

    def __init__(self):
        self.trades_list = []
        self.entry = None
        self.equity_curve, self.equity_dates = [], []
        self.bars_in_market = 0

    def next(self):
        i = len(self) - 1
        sig = self.params.signals[i]

        if sig == 1 and not self.position:
            price = self.data.close[0]
            size = self.broker.getcash() / (price * (1 + self.params.commission)) * 0.999
            if size > 0:
                self.buy(size=size)
                conf = self.params.confidence[i] if self.params.confidence is not None else np.nan
                self.entry = {"idx": i, "date": self.data.datetime.date(0), "price": price, "prob": conf}
        elif sig == 0 and self.position:
            self.close()
            self._record_exit(self.data.close[0], self.data.datetime.date(0))

        if self.entry is not None:  # long from the entry bar's close up to the exit bar
            self.bars_in_market += 1
        # Value at this bar's close (orders placed now fill at this close on the next broker tick)
        self.equity_curve.append(self.broker.getvalue())
        self.equity_dates.append(self.data.datetime.date(0))

    def _record_exit(self, price, date):
        if self.entry is None:
            return
        e = self.entry
        closes = self.params.closes
        h = e["idx"] + HORIZON
        actual_up = int(closes[h] > e["price"]) if h < len(closes) else int(price > e["price"])
        self.trades_list.append({
            "entry_date": e["date"],
            "exit_date": date,
            "entry_price": e["price"],
            "exit_price": price,
            "direction": "LONG",
            "prediction": "UP",
            "return": price / e["price"] - 1,
            "actual_direction": actual_up,
            "correct": actual_up,            # was price higher HORIZON bars after entry?
            "confidence": e["prob"],
            "open_at_end": False,
        })
        self.entry = None

    def stop(self):
        if self.position and self.entry is not None:
            self._record_exit(self.data.close[0], self.data.datetime.date(0))
            self.trades_list[-1]["open_at_end"] = True


def _run_engine(df_bt: pd.DataFrame, signals: np.ndarray, initial_capital: float,
                commission: float, confidence: Optional[np.ndarray] = None) -> SignalStrategy:
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(bt.feeds.PandasData(dataname=df_bt, openinterest=-1))
    cerebro.addstrategy(
        SignalStrategy,
        signals=np.asarray(signals),
        confidence=confidence,
        closes=df_bt["close"].values,
        commission=commission,
    )
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_coc(True)  # fill market orders at the signal bar's close
    strat = cerebro.run()[0]
    strat.final_value = cerebro.broker.getvalue()
    return strat


def _sharpe(equity: np.ndarray, periods_per_year: int) -> float:
    if len(equity) < 3:
        return 0.0
    rets = np.diff(equity) / equity[:-1]
    if rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(periods_per_year))


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(((peak - equity) / peak).max())


def _summarize(name: str, strat: SignalStrategy, initial_capital: float, periods: int) -> Dict:
    equity = np.asarray(strat.equity_curve, dtype=float)
    trades = pd.DataFrame(strat.trades_list)
    n = len(trades)
    wins = trades[trades["return"] > 0] if n else trades
    return {
        "name": name,
        "total_return": strat.final_value / initial_capital - 1,
        "final_capital": strat.final_value,
        "sharpe_ratio": _sharpe(equity, periods),
        "max_drawdown": _max_drawdown(equity),
        "total_trades": n,
        "win_rate": len(wins) / n if n else 0.0,
        "time_in_market": strat.bars_in_market / max(len(equity), 1),
        "equity_curve": equity.tolist(),
    }


# ==========================================================
# PUBLIC API
# ==========================================================
def run_backtest(
    symbol: str,
    model_path: str = "artifacts/general_crypto_model.pkl",
    feature_cols_path: str = "artifacts/feature_columns.pkl",
    lookback: str = "5y",
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    confidence_threshold: float = 0.5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    out_of_sample: bool = True,
    include_benchmarks: bool = True,
    n_shuffles: int = N_SHUFFLES,
) -> Dict:
    """
    Run the ML backtest (and, by default, benchmarks + shuffle test).
    Returns a dict of metrics, trades and curves, or {"error": ...}.
    With out_of_sample=True the window starts after the model's training period.
    """
    try:
        model = joblib.load(model_path)
        feature_cols = joblib.load(feature_cols_path)
    except FileNotFoundError:
        return {"error": "Model files not found. Run `python -m src.train_model` first."}

    try:
        raw = load_crypto(symbol, lookback=lookback)
    except DataLoadError:
        if "USDT" in symbol.upper():
            suggested = symbol.upper().replace("-USDT", "-USD")
            return {"error": f"Symbol '{symbol}' not found. Try '{suggested}' instead (e.g., BTC-USD, ETH-USD)"}
        return {"error": f"Could not load data for '{symbol}'. Try: BTC-USD, ETH-USD, SOL-USD, AAPL, TSLA, or other valid symbols."}
    except Exception as e:
        return {"error": f"Error loading data for '{symbol}': {e}"}

    if raw.empty:
        return {"error": f"No data available for symbol '{symbol}'. Try a different symbol or longer lookback period."}

    full = add_indicators(raw)
    if full.empty:
        return {"error": "No data after adding indicators"}
    missing = [c for c in feature_cols if c not in full.columns]
    if missing:
        return {"error": f"Saved model expects features {missing} that are not computed. Retrain with `python -m src.train_model`."}

    bench_full = benchmark_signals(full) if include_benchmarks else {}

    # ---- choose the test window ----
    meta = load_training_metadata()
    oos_start = None
    if out_of_sample and meta:
        oos_start = pd.Timestamp(meta["train_end"]) + pd.Timedelta(days=meta.get("horizon", HORIZON) + 1)
    df = full
    starts = [pd.Timestamp(d) for d in (start_date, oos_start) if d is not None]
    if starts:
        df = df[df.index >= max(starts)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    if len(df) < 30:
        hint = ""
        if oos_start is not None:
            hint = (f" The model was trained on data up to {meta['train_end']}, so the out-of-sample "
                    f"window starts {oos_start.date()}. Use a longer lookback, untick out-of-sample, or retrain.")
        return {"error": f"Not enough data for backtesting ({len(df)} bars).{hint}"}

    in_sample_overlap = bool(
        meta and symbol.upper() in [s.upper() for s in meta.get("symbols", [])]
        and df.index.min() <= pd.Timestamp(meta["train_end"])
    )

    df_bt = df[["open", "high", "low", "close", "volume"]].copy()
    df_bt["volume"] = df_bt["volume"].fillna(0)
    periods = 365 if symbol.upper().endswith("-USD") else 252

    # ---- ML strategy ----
    feats = df[feature_cols]
    valid = feats.notna().all(axis=1).values
    probs = np.full(len(df), np.nan)
    if valid.any():
        probs[valid] = model.predict_proba(feats[valid])[:, 1]
    buy_th, sell_th = _thresholds(confidence_threshold)
    sig_ml = ml_signals(probs, buy_th, sell_th)

    try:
        strat = _run_engine(df_bt, sig_ml, initial_capital, commission, confidence=probs)
    except Exception as e:
        return {"error": f"Backtest execution failed: {e}"}

    buy_n, sell_n = int((sig_ml == 1).sum()), int((sig_ml == 0).sum())
    neutral_n = int(((sig_ml == -1) & valid).sum())
    trades_df = pd.DataFrame(strat.trades_list)

    if trades_df.empty:
        return {
            "error": "No trades executed",
            "symbol": symbol,
            "initial_capital": initial_capital,
            "final_capital": strat.final_value,
            "diagnostics": {
                "buy_signals": buy_n,
                "sell_signals": sell_n,
                "neutral_signals": neutral_n,
                "total_predictions_processed": buy_n + sell_n + neutral_n,
                "confidence_threshold": confidence_threshold,
                "data_points_available": len(df),
            },
            "suggestion": "Try lowering the confidence threshold or using a longer lookback period.",
        }

    ml_summary = _summarize(ML_NAME, strat, initial_capital, periods)
    equity = np.asarray(strat.equity_curve, dtype=float)
    dates = [str(d) for d in strat.equity_dates]
    closes = df["close"].values[: len(equity)]
    buy_hold_curve = (initial_capital * closes / closes[0]).tolist()
    buy_hold_return = closes[-1] / closes[0] - 1

    # ---- benchmarks (same window, engine, capital, commission) ----
    benchmarks: List[Dict] = []
    for name, sig in bench_full.items():
        b = _summarize(name, _run_engine(df_bt, sig.loc[df.index].values, initial_capital, commission),
                       initial_capital, periods)
        b["rule"] = BENCHMARK_RULES.get(name, "")
        benchmarks.append(b)

    # ---- shuffle test: same signal mix, random timing ----
    shuffle = None
    if include_benchmarks and n_shuffles > 0:
        rng = np.random.default_rng(SHUFFLE_SEED)
        sig_valid = sig_ml[valid]
        shuffled_returns = []
        for _ in range(n_shuffles):
            s = sig_ml.copy()
            s[valid] = rng.permutation(sig_valid)
            st = _run_engine(df_bt, s, initial_capital, commission)
            shuffled_returns.append(st.final_value / initial_capital - 1)
        shuffled_returns = np.array(shuffled_returns)
        shuffle = {
            "n": n_shuffles,
            "median_return": float(np.median(shuffled_returns)),
            "p10_return": float(np.percentile(shuffled_returns, 10)),
            "p90_return": float(np.percentile(shuffled_returns, 90)),
            "ml_beats_pct": float((ml_summary["total_return"] > shuffled_returns).mean()),
            "returns": shuffled_returns.tolist(),
        }

    # ---- ML trade metrics ----
    wins = trades_df[trades_df["return"] > 0]
    losses = trades_df[trades_df["return"] <= 0]
    loss_sum = losses["return"].sum()
    total_return = ml_summary["total_return"]

    return {
        "symbol": symbol,
        "initial_capital": initial_capital,
        "final_capital": strat.final_value,
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "excess_return": total_return - buy_hold_return,
        "total_trades": len(trades_df),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(trades_df),
        "avg_win": wins["return"].mean() if len(wins) else 0.0,
        "avg_loss": losses["return"].mean() if len(losses) else 0.0,
        "profit_factor": abs(wins["return"].sum() / loss_sum) if loss_sum != 0 else 0.0,
        "prediction_accuracy": trades_df["correct"].mean(),
        "max_drawdown": ml_summary["max_drawdown"],
        "sharpe_ratio": ml_summary["sharpe_ratio"],
        "time_in_market": ml_summary["time_in_market"],
        "trades": trades_df.to_dict("records"),
        "equity_curve": equity.tolist(),
        "buy_hold_curve": buy_hold_curve,
        "dates": dates,
        "period_start": dates[0],
        "period_end": dates[-1],
        "out_of_sample": oos_start is not None,
        "in_sample_overlap": in_sample_overlap,
        "buy_threshold": buy_th,
        "sell_threshold": sell_th,
        "ml_summary": ml_summary,
        "benchmarks": benchmarks,
        "shuffle_test": shuffle,
    }
