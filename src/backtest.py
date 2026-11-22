"""
backtest.py
Backtest (with backtrader) a long-only strategy that trades on the ML model's predictions.

Key points
- Indicators are computed on the full history first, then the test window is sliced,
  so the window starts with valid indicators (no warm-up gap).
- By default the backtest only covers dates AFTER the model's training period
  (out_of_sample=True), so results are not inflated by data the model has seen.
- Orders fill at the close of the signal bar (the features are known at that close),
  and position size is fractional, so expensive assets like BTC trade with small capital.
"""

from typing import Dict, Optional

import backtrader as bt
import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_crypto, DataLoadError
from src.features import HORIZON, add_indicators

METADATA_PATH = "artifacts/training_metadata.pkl"


def load_training_metadata(path: str = METADATA_PATH) -> Optional[dict]:
    try:
        return joblib.load(path)
    except (FileNotFoundError, OSError):
        return None


def _thresholds(confidence_threshold: float):
    """Map the 0-1 UI slider to buy/sell probability thresholds (asymmetric)."""
    delta = 0.01 + confidence_threshold * 0.09
    return 0.5 + delta, 0.5 - delta * 0.7


class MLStrategy(bt.Strategy):
    """Long-only: buy when P(up) > buy_threshold, exit when P(up) < sell_threshold."""

    params = (
        ("model", None),
        ("feature_cols", None),
        ("features", None),        # DataFrame aligned 1:1 with the data feed
        ("confidence_threshold", 0.5),
        ("commission", 0.001),
    )

    def __init__(self):
        p = self.params
        self.buy_threshold, self.sell_threshold = _thresholds(p.confidence_threshold)
        feats = p.features[p.feature_cols]
        valid = feats.notna().all(axis=1)
        probs = np.full(len(feats), np.nan)
        if valid.any():
            probs[valid.values] = p.model.predict_proba(feats[valid])[:, 1]
        self.probs = probs  # precomputed once: same result as per-bar predict, much faster

        self.buy_signals = self.sell_signals = self.neutral_signals = self.skipped = 0
        self.trades_list = []
        self.entry = None  # dict with bar index, date, price, prob
        self.equity_curve, self.equity_dates = [], []

    def next(self):
        i = len(self) - 1
        prob = self.probs[i]

        if np.isnan(prob):
            self.skipped += 1
        elif prob > self.buy_threshold:
            self.buy_signals += 1
            if not self.position:
                price = self.data.close[0]
                cash = self.broker.getcash()
                size = cash / (price * (1 + self.params.commission)) * 0.999
                if size > 0:
                    self.buy(size=size)
                    self.entry = {"idx": i, "date": self.data.datetime.date(0), "price": price, "prob": prob}
        elif prob < self.sell_threshold:
            self.sell_signals += 1
            if self.position:
                self.close()
                self._record_exit(i, self.data.close[0], self.data.datetime.date(0))
        else:
            self.neutral_signals += 1

        # Value at this bar's close (orders placed now fill at this close on the next broker tick)
        self.equity_curve.append(self.broker.getvalue())
        self.equity_dates.append(self.data.datetime.date(0))

    def _record_exit(self, i, price, date):
        if self.entry is None:
            return
        e = self.entry
        closes = self.params.features["close"].values
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
            "correct": actual_up,            # was the UP call right over the model's horizon?
            "confidence": e["prob"],
            "open_at_end": False,
        })
        self.entry = None

    def stop(self):
        if self.position and self.entry is not None:
            n = len(self) - 1
            self._record_exit(n, self.data.close[0], self.data.datetime.date(0))
            self.trades_list[-1]["open_at_end"] = True


def _sharpe(equity: np.ndarray, periods_per_year: int) -> float:
    rets = np.diff(equity) / equity[:-1]
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(periods_per_year))


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(((peak - equity) / peak).max()) if len(equity) else 0.0


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
) -> Dict:
    """
    Run the backtest. Returns a dict of metrics, trades and curves, or {"error": ...}.
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

    df = add_indicators(raw)
    if df.empty:
        return {"error": "No data after adding indicators"}
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return {"error": f"Saved model expects features {missing} that are not computed. Retrain with `python -m src.train_model`."}

    # ---- choose the test window ----
    meta = load_training_metadata()
    oos_start = None
    if out_of_sample and meta:
        oos_start = pd.Timestamp(meta["train_end"]) + pd.Timedelta(days=meta.get("horizon", HORIZON) + 1)
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

    # ---- run backtrader ----
    df_bt = df[["open", "high", "low", "close", "volume"]].copy()
    df_bt["volume"] = df_bt["volume"].fillna(0)
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(bt.feeds.PandasData(dataname=df_bt, openinterest=-1))
    cerebro.addstrategy(
        MLStrategy,
        model=model,
        feature_cols=feature_cols,
        features=df,
        confidence_threshold=confidence_threshold,
        commission=commission,
    )
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_coc(True)  # fill market orders at the signal bar's close

    try:
        strat = cerebro.run()[0]
    except Exception as e:
        return {"error": f"Backtest execution failed: {e}"}

    final_value = cerebro.broker.getvalue()
    trades_df = pd.DataFrame(strat.trades_list)

    if trades_df.empty:
        return {
            "error": "No trades executed",
            "symbol": symbol,
            "initial_capital": initial_capital,
            "final_capital": final_value,
            "diagnostics": {
                "buy_signals": strat.buy_signals,
                "sell_signals": strat.sell_signals,
                "neutral_signals": strat.neutral_signals,
                "total_predictions_processed": strat.buy_signals + strat.sell_signals + strat.neutral_signals,
                "confidence_threshold": confidence_threshold,
                "data_points_available": len(df),
            },
            "suggestion": "Try lowering the confidence threshold or using a longer lookback period.",
        }

    # ---- metrics ----
    equity = np.asarray(strat.equity_curve, dtype=float)
    dates = [str(d) for d in strat.equity_dates]
    closes = df["close"].values[: len(equity)]
    buy_hold_curve = (initial_capital * closes / closes[0]).tolist()

    periods = 365 if symbol.upper().endswith("-USD") else 252
    total_return = final_value / initial_capital - 1
    buy_hold_return = closes[-1] / closes[0] - 1

    wins = trades_df[trades_df["return"] > 0]
    losses = trades_df[trades_df["return"] <= 0]
    loss_sum = losses["return"].sum()

    return {
        "symbol": symbol,
        "initial_capital": initial_capital,
        "final_capital": final_value,
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
        "max_drawdown": _max_drawdown(equity),
        "sharpe_ratio": _sharpe(equity, periods),
        "trades": trades_df.to_dict("records"),
        "equity_curve": equity.tolist(),
        "buy_hold_curve": buy_hold_curve,
        "dates": dates,
        "period_start": dates[0],
        "period_end": dates[-1],
        "out_of_sample": oos_start is not None,
        "in_sample_overlap": in_sample_overlap,
        "buy_threshold": strat.buy_threshold,
        "sell_threshold": strat.sell_threshold,
    }
