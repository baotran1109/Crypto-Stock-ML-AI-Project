# Crypto Stock ML AI Project

A machine learning project that predicts short-term price direction for cryptocurrencies and stocks, then backtests a simple trading strategy and exposes everything through a web dashboard.

## What It Does

- Loads OHLCV data for crypto and stocks from Yahoo Finance, with CoinGecko as a fallback for some coins.
- Builds technical indicators (RSI, EMAs, MACD, Bollinger Bands, returns, volatility) and a supervised dataset that predicts whether price will be higher in 5 days.
- Trains a Random Forest classifier on multiple symbols, evaluates it with time-series cross-validation, and saves the model and metrics.
- Backtests a strategy that buys and sells based on model confidence using the Backtrader framework.
- Provides a Streamlit app with three pages: live prediction, model performance metrics, and backtesting.

## Project Structure

- **src/data_loader.py** – Loads OHLCV from Yahoo Finance (and CoinGecko when needed). Caches data as CSV under the data folder. Normalizes column names and timezone.
- **src/features.py** – Adds technical indicators and builds the supervised dataset (5-day ahead up/down target).
- **src/train_model.py** – Trains the model on several symbols, runs time-series CV, saves the model, feature list, evaluation metrics, and feature importance to the artifacts folder.
- **src/backtest.py** – Runs a Backtrader backtest where the strategy trades on model predictions; returns returns, win rate, Sharpe ratio, drawdown, equity curve, and trade list.
- **app/streamlit_app.py** – Streamlit UI: Prediction (symbol and chart), Model Performance (metrics and feature importance), Backtest (parameters and results).
- **data/** – Cached CSV files per symbol, source, and lookback (e.g. BTC_USD yahoo 5y).
- **artifacts/** – Saved model (general_crypto_model.pkl), feature columns, evaluation metrics, feature importance.

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
- **Backtest:** Pick a symbol, confidence threshold, initial capital, commission, and lookback; run the backtest to see total return, win rate, Sharpe ratio, max drawdown, equity curve, and trade list.

## Data and Model

- Data is cached under **data/** with filenames like SYMBOL__source__lookback.csv. Yahoo is tried first; CoinGecko is used as fallback for supported coins (e.g. BTC, ETH, SOL).
- The model predicts a binary target: whether close price in 5 days is above current close (1) or not (0). Features include RSI, EMAs, MACD, Bollinger Bands, returns, volatility, and derived ratios.
- Artifacts are written to **artifacts/**: the trained model, feature column list, evaluation metrics, and feature importance. The Streamlit app and backtest both rely on these files.

## Notes

- Use symbol format BTC-USD, ETH-USD for crypto (not BTC-USDT). Use ticker symbols like AAPL, TSLA for stocks.
- If the app says model files not found, run `python -m src.train_model` first.
- Backtest needs enough history (e.g. 5y) and may produce no trades if the confidence threshold is too high; try lowering it or using a longer lookback.
