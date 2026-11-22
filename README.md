# Crypto Stock ML AI Project

A machine learning project that predicts short-term price direction for cryptocurrencies and stocks, backtests trading strategies honestly against simple benchmarks, and exposes everything through a web dashboard.

## What It Does

- Loads OHLCV data for crypto and stocks from Yahoo Finance, with CoinGecko as a fallback for some coins.
- Builds scale-free technical features (RSI, price vs EMAs, MACD as % of price, Bollinger Band position/width, returns, volatility) and a supervised dataset that predicts whether price will be higher in 5 days.
- Trains **one model per asset class**: a crypto model on 10 coins and a stock model on 10 large stocks/ETFs. The app picks the right one for each symbol (`-USD` tickers use the crypto model).
- Splits train/test by date (with a gap so labels never overlap), evaluates with walk-forward cross-validation, and saves the model, metrics and training dates.
- Backtests a long-only strategy that trades on model confidence using the Backtrader framework, by default only on dates after the training period, and compares it against rule-based benchmarks and a shuffle test. Optional risk rules: volatility-based sizing, minimum holding period, slippage.
- Includes a signal research harness that tests many targets, feature sets and models walk-forward, plus a crypto rotation simulation.
- Provides a Streamlit app with four pages: Prediction, Model Performance, Backtest, and Research.

## Project Structure

- **src/data_loader.py** – Loads OHLCV from Yahoo Finance (and CoinGecko when needed). Caches data as CSV under the data folder. Normalizes column names and timezone.
- **src/features.py** – Adds technical indicators and builds the supervised dataset (5-day ahead up/down target). Only scale-free columns in `FEATURE_COLS` are model inputs; raw dollar indicators are kept for charts.
- **src/model_registry.py** – Training universes per asset class, `asset_class_for(symbol)`, and loading each class's artifacts.
- **src/train_model.py** – Trains the crypto and stock models with a date-based split and walk-forward CV; saves each to `artifacts/<asset_class>/`.
- **src/backtest.py** – One signal-driven Backtrader engine for the ML strategy, the benchmarks and the shuffle test, with optional risk rules.
- **src/research.py** – Signal research harness (walk-forward grid over targets × feature sets × models, and a crypto rotation simulation). Writes `reports/`.
- **app/streamlit_app.py** – Streamlit UI.
- **data/** – Cached CSV files per symbol, source, and lookback (e.g. BTC_USD yahoo 5y).
- **artifacts/crypto/**, **artifacts/stock/** – `model.pkl`, feature columns, evaluation metrics, feature importance, training metadata.
- **reports/** – `FINDINGS.md` (summary), `signal_research.md` / `.csv` (every experiment), `rotation_research.csv`.

## Requirements

- Python 3 with packages listed in requirements.txt: streamlit, pandas, numpy, scikit-learn, joblib, yfinance, requests, urllib3, plotly, backtrader.

## Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies: `pip install -r requirements.txt`
3. Train the models once: `python -m src.train_model` (from the project root; needs network or existing cached data in data/).
4. Run the app: `streamlit run app/streamlit_app.py` (from the project root).

## Usage

- **Training:** `python -m src.train_model` trains both models; `python -m src.train_model --classes stock` trains one. Set FORCE_REFRESH in the script to pull fresh data.
- **Research:** `python -m src.research` runs the full experiment grid and rotation simulation (about 5 minutes; downloads 10 years of data the first time). `--quick` skips the Random Forest, `--no-rotation` skips the rotation simulation.
- **Dashboard:** `streamlit run app/streamlit_app.py`, then use the sidebar to switch pages.
- **Prediction:** Enter a symbol (e.g. BTC-USD, ETH-USD, AAPL) and click Predict to see UP/DOWN, P(up), which model was used, and a chart.
- **Model Performance:** Pick the crypto or stock model to see its test metrics, walk-forward CV, per-symbol results and feature importance.
- **Backtest:** Pick a symbol, confidence threshold, capital, commission, lookback and optional risk rules; see the ML strategy against benchmarks and the shuffle test.
- **Research:** Shows `reports/FINDINGS.md` and the full experiment tables.

## Data and Model

- Data is cached under **data/** with filenames like SYMBOL__source__lookback.csv. Yahoo is tried first; CoinGecko is used as fallback for supported coins (e.g. BTC, ETH, SOL). A cache older than 12 hours is re-downloaded; if the download fails, the stale cache is used.
- The models predict a binary target: whether close price in 5 days is above current close (1) or not (0). The last 5 days of history have no known outcome and are excluded from training. Features use only data up to the decision bar's close, and live prediction uses the latest bar the same way.
- Crypto and stocks get separate models because they behave differently: crypto trades 24/7 and is several times more volatile, stocks trade weekdays and react to earnings, rates and the overall market.
- A symbol outside a model's training list (e.g. TSM) still uses its asset class's model; the app notes that the model never saw that symbol's history.

## Benchmarks

Every backtest also runs these rules through the same engine, window, capital, commission and risk rules, so the comparison is fair:

| Benchmark | Rule |
|---|---|
| Buy & hold | Buy on the first day and hold to the end. |
| SMA 50/200 crossover | Long while the 50-day average is above the 200-day average. |
| Momentum (20-day) | Long while the price is above its level 20 days ago. |
| RSI mean reversion | Buy when RSI drops below 30, sell when it rises above 55. |

**Shuffle test:** the ML model's own buy/sell/hold signals are randomly reordered 20 times, keeping the same number of each signal but with random timing. If the model's timing has real skill, it should beat most shuffles (say 90% or more). Around 50% means its timing is no better than chance.

## Risk Rules (optional, Backtest page)

- **Volatility-based sizing:** position size = target volatility ÷ current annualized volatility (20-day), capped at 100%. Invests less when the market is wild.
- **Minimum holding period:** exit signals are ignored until a position has been held this many bars, cutting back-and-forth trading.
- **Slippage:** extra cost per trade side on top of commission.

These apply to every strategy in a run, not just the ML model.

## Research Findings (short version)

See `reports/FINDINGS.md` for detail. Across 100+ walk-forward experiments:

- **Price direction (up/down in 5 or 20 days) is not predictable** from these features, for crypto or stocks. Test ROC-AUC stays around 0.50.
- **Volatility is predictable**: whether the next 5 days will have a big move scores ROC-AUC ~0.55 (crypto) and ~0.58 (stocks), in every test period. That supports volatility-based sizing, not direction calls.
- **Which coin will beat the other coins** over 5 days is weakly predictable (ROC-AUC ~0.52–0.53, every test period), but after trading costs a rotation strategy built on it does not reliably beat holding all coins, and a plain momentum rule beat it in most configurations.

## Notes

- Use symbol format BTC-USD, ETH-USD for crypto (not BTC-USDT). Use ticker symbols like AAPL, TSLA for stocks.
- If the app says model files not found, run `python -m src.train_model` first.
- The backtest is out-of-sample by default: it starts after the model's training period. Unticking that option includes training data and makes results look much better than they are.
- Orders fill at the signal bar's close with fractional position sizes, so expensive assets like BTC can be traded with small capital.
- A single backtest on one symbol is noisy. The same model can beat every benchmark on one stock and lose on the next; check the shuffle test and several symbols before reading anything into it.
