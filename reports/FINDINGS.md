# Signal Research Findings

**Bottom line:** none of the signals found is strong enough to trade on after costs. Price direction is not predictable from these features. Volatility is. A weak "which coin beats the others" signal exists but loses to simply holding all coins in most setups.

Generated from `python -m src.research` (104 walk-forward experiments + 13 rotation simulations, 10 years of daily data). Full tables: `signal_research.md`.

## How everything was tested

- **Universes:** 10 coins (BTC, ETH, SOL, XRP, ADA, DOGE, LINK, LTC, BNB, AVAX) and 10 large stocks/ETFs (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, SPY, QQQ, DIA), researched separately.
- **Walk-forward by date:** the last 60% of dates (mid-2021 to Nov 2026) is cut into 6 test periods. Each period is predicted by a model trained only on earlier dates, with a gap so no training label overlaps it.
- **Scored by ROC-AUC** (0.5 = random) per test period. A result only counts if it beats 0.5 in all 6 periods, not just on average.
- **Every target has a no-ML rule baseline** to beat.

## What was tried

| | Options |
|---|---|
| Targets | up/down in 5 days, up/down in 20 days, beat the median coin/stock over 5 days, big move (either way) in 5 days |
| Feature sets | base indicators → + longer-term price → + volume → + cross-market (BTC, S&P 500, VIX, 10-year yield, dollar) → + cross-sectional ranks and day of week |
| Models | logistic regression, gradient boosting, random forest |

## Results

| Question | Crypto ROC-AUC | Stocks ROC-AUC | Holds in every test period? | Verdict |
|---|---|---|---|---|
| Up or down in 5 days? | 0.51 (best 0.52) | 0.51 (best 0.52) | No | **Not predictable** |
| Up or down in 20 days? | 0.52 (best 0.55) | 0.51 (best 0.54) | No (best: 5 of 6) | **Not reliable** |
| Beat the other coins/stocks over 5 days? | 0.53 (best 0.54) | 0.51 | Crypto: yes, in most setups. Stocks: no | **Weak but real for crypto** |
| Big move in the next 5 days? | 0.54 (best 0.55) | 0.58 (best 0.58) | Yes, in most setups | **Predictable** |

Extra features helped volatility and relative-performance predictions a little. They did not rescue direction predictions.

## Can the crypto "beat the other coins" signal make money?

Rotation test: every N days, hold the k coins the model ranks highest, equal weight, paying trading costs. 13 configurations were tested and **all are reported**, not just the good ones (`signal_research.md`, last table).

- The ML picks beat **random picks** in 78–100% of runs, so the ranking carries information.
- The ML rotation beat the **equal-weight basket** (just hold all 10 coins) in only **5 of 13** configurations.
- A plain **20-day momentum rule** (hold the coins that rose most) beat the ML rotation in **12 of 13**.
- At 0.30% cost per trade side, every 5-day ML rotation lost money.
- Every strategy, including the basket, had drawdowns of 70–85%.

The signal is too small compared to trading costs, and results swing heavily with the holding period and number of coins. That's a sign of noise, not an edge.

## Single-symbol backtests are noisy

Out-of-sample backtests of the stock model (Nov 2025 to Nov 2026), share of random-timing shuffles beaten:

| MSFT | AAPL | GOOGL | SPY | NVDA |
|---|---|---|---|---|
| 98% | 72% | 26% | 6% | 0% |

The same model looks brilliant on MSFT and useless on NVDA. One good backtest proves nothing.

## What this means for the project

1. **The app's up/down predictions are no better than a coin flip** for both asset classes. The Model Performance page shows this.
2. **Volatility-based position sizing** (Backtest page → Risk rules) is the refinement supported by the evidence. Volatility is predictable even when direction is not.
3. **No confidence-based sizing or stop-loss tuning was added.** Tuning these on a random signal would only fit noise.
4. **Ideas not yet tried**, which need other data sources:
   - Crypto funding rates and open interest
   - On-chain flows
   - Order-book data
   - News or sentiment
   - Intraday data
   - For stocks: earnings dates, analyst revisions, valuation

## Caveat on multiple testing

Testing 104 combinations means a few will look good by chance. That's why results are judged by consistency across all 6 test periods and across models, not by the single best score. The 20-day crypto result (best 0.552, 5 of 6 periods) is exactly the kind of lone good number that should not be trusted without new data.
