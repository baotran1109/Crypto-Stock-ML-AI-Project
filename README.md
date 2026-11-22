# Crypto Stock ML AI Project

A machine learning project that predicts short-term price direction for cryptocurrencies and stocks, then backtests a simple trading strategy and exposes everything through a web dashboard.

## What It Does

- Loads OHLCV data for crypto and stocks from Yahoo Finance, with CoinGecko as a fallback for some coins.
- Builds scale-free technical features (RSI, price vs EMAs, MACD as % of price, Bollinger Band position/width, returns, volatility) and a supervised dataset that predicts whether price will be higher in 5 days.
- Trains one Random Forest classifier on several coins, splits train/test by date (with a gap so labels never overlap), evaluates it with walk-forward cross-validation, and saves the model, metrics and training dates.
- Backtests a long-only strategy that trades on model confidence using the Backtrader framework, by default only on dates after the training period, and compares it against simple rule-based benchmarks and a shuffle test.
- Provides a Streamlit app with three pages: live prediction, model performance metrics, and backtesting.

## Project Structure

- **src/data_loader.py** – Loads OHLCV from Yahoo Finance (and CoinGecko when needed). Caches data as CSV under the data folder. Normalizes column names and timezone.
- **src/features.py** – Adds technical indicators and builds the supervised dataset (5-day ahead up/down target). Only scale-free columns in `FEATURE_COLS` are model inputs; raw dollar indicators are kept for charts.
- **src/train_model.py** – Trains the model on several symbols with a date-based split and walk-forward CV; saves the model, feature list, evaluation metrics, feature importance, and training metadata (train/test dates) to the artifacts folder.
- **src/backtest.py** – Runs a Backtrader backtest where the strategy trades on model predictions; returns returns, win rate, Sharpe ratio, drawdown, equity curve, and trade list. Also runs the benchmarks and the shuffle test through the same engine.
- **app/streamlit_app.py** – Streamlit UI: Prediction (symbol and chart), Model Performance (metrics and feature importance), Backtest (parameters and results).
- **data/** – Cached CSV files per symbol, source, and lookback (e.g. BTC_USD yahoo 5y).
- **artifacts/** – Saved model (general_crypto_model.pkl), feature columns, evaluation metrics, feature importance, training metadata.

## Requirements

- Python 3 with packages listed in requirements.txt: streamlit, pandas, numpy, scikit-learn, joblib, yfinance, requests, urllib3, plotly, backtrader.

## Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model once: `python -m src.train_model` (from the project root; needs network or existing cached data in data/).
4. Run the app: `streamlit run app/streamlit_app.py` (from the project root).

## Usage

- **Training:** Run `python -m src.train_model` to fetch data, add indicators, train the model, and write artifacts. You can set FORCE_REFRESH in the script to pull fresh data from the API.
- **Dashboard:** Run `streamlit run app/streamlit_app.py` and open the URL in a browser. Use the sidebar to switch between Prediction, Model Performance, and Backtest.
- **Prediction:** Enter a symbol (e.g. BTC-USD, ETH-USD, AAPL), click Predict to see UP/DOWN and confidence, plus optional candlestick and volume chart.
- **Backtest:** Pick a symbol, confidence threshold, initial capital, commission, and lookback; run the backtest to see total return, win rate, Sharpe ratio, max drawdown, equity curve, trade list, a benchmark comparison table, and the shuffle test.

## Data and Model

- Data is cached under **data/** with filenames like SYMBOL__source__lookback.csv. Yahoo is tried first; CoinGecko is used as fallback for supported coins (e.g. BTC, ETH, SOL). A cache older than 12 hours is re-downloaded; if the download fails, the stale cache is used.
- The model predicts a binary target: whether close price in 5 days is above current close (1) or not (0). The last 5 days of history have no known outcome and are excluded from training. Features use only data up to the decision bar's close, and live prediction uses the latest bar the same way.
- Because features are ratios rather than dollar values, the same model can be applied to stocks, but it was trained on crypto only, so stock predictions are out of domain.
- Artifacts are written to **artifacts/**: the trained model, feature column list, evaluation metrics, and feature importance. The Streamlit app and backtest both rely on these files.

## Benchmarks

Every backtest also runs these rules through the same engine, window, capital and commission, so the comparison is fair:

| Benchmark | Rule |
|---|---|
| Buy & hold | Buy on the first day and hold to the end. |
| SMA 50/200 crossover | Long while the 50-day average is above the 200-day average. |
| Momentum (20-day) | Long while the price is above its level 20 days ago. |
| RSI mean reversion | Buy when RSI drops below 30, sell when it rises above 55. |

**Shuffle test:** the ML model's own buy/sell/hold signals are randomly reordered 20 times, keeping the same number of each signal but with random timing. If the model's timing has real skill, it should beat most shuffles (say 90% or more). Around 50% means its timing is no better than chance.

## Notes

- Use symbol format BTC-USD, ETH-USD for crypto (not BTC-USDT). Use ticker symbols like AAPL, TSLA for stocks.
- If the app says model files not found, run `python -m src.train_model` first.
- The backtest is out-of-sample by default: it starts after the model's training period. Unticking that option includes training data and makes results look much better than they are.
- Orders fill at the signal bar's close with fractional position sizes, so expensive assets like BTC can be traded with small capital.
- Honest expectation: with daily technical indicators alone, the model's out-of-sample ROC-AUC is about 0.50 (no better than random). Check the Model Performance page before reading anything into backtest results.
