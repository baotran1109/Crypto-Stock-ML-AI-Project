# Signal Research Findings

**Bottom line:** none of the signals found is strong enough to trade on after costs. Price direction is not predictable from these features. Volatility is. A weak "which coin beats the others" signal exists but loses to simply holding all coins in most setups. Adding non-price data (funding rates, Fear & Greed, earnings, analyst ratings, macro) did not change that; see [Round 2](#round-2-non-price-data). On a 99-stock / 39-coin universe, the round-2 stock lead disappeared and crypto ranking stayed weak but consistent. A crypto rotation beat holding all coins in every configuration, but the coin list was chosen with hindsight; see [Round 3](#round-3-large-universe-99-stocks-39-coins).

Generated from `python -m src.research` (172 walk-forward experiments + 13 rotation simulations, 10 years of daily data). Full tables: `signal_research.md`.

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
4. **Tried in round 2 with no direction edge:** funding rates, Fear & Greed, earnings surprises, analyst actions, yield curve and credit spreads.
5. **Ideas not yet tried:**
   - A point-in-time crypto universe (top coins by market cap as of each date) to retest the round-3 rotation without hindsight
   - Open interest and liquidations (no free long history from this region)
   - On-chain flows
   - Order-book data
   - News sentiment
   - Intraday data
   - Company fundamentals from SEC EDGAR (needs a contact email in requests)

## Round 2: non-price data

Loaded by `src/alt_data.py` from free sources and cached in `data/alt/`. Binance and Bybit block this region, so Hyperliquid and Deribit supply the funding rates.

| Data | Source | History | Feature set |
|---|---|---|---|
| Fear & Greed index, BTC funding rate | alternative.me, Deribit | 2018 / 2019+ | crypto `+sentiment` |
| Per-coin funding rate, all 10 coins | Hyperliquid | May 2023+ | crypto `+funding` |
| Yield curve (10y-2y), corporate credit spread | FRED | decades | stock `+macro` |
| Earnings surprise, days since earnings, analyst upgrades/downgrades and price-target changes | yfinance | 2005+ / 2012+ | stock `+events` (7 companies; ETFs have none) |

Each value is only used once it was public: funding and Fear & Greed from the end of their day, earnings and analyst actions and FRED data from the next day.

Per-coin funding only goes back to mid-2023, so it was tested in a separate **2023+ window** where `+cross`, `+sentiment` and `+funding` are compared on the same dates (6 test periods of about 4 months).

| Question | Before (`+cross`) | With new data | Verdict |
|---|---|---|---|
| Crypto up/down in 5 days | 0.51 | 0.51 (`+sentiment`), 0.52 (`+funding`, 2023+) | No change |
| Crypto beat the other coins | 0.53 | 0.53–0.54 (`+sentiment`); 2023+: 0.566 → 0.563 with funding | No change |
| Crypto big move | 0.55 | 0.55–0.56 (`+sentiment`) | Slightly better |
| Stock up/down in 5 days | 0.51 | 0.51 (`+macro`, `+events`) | No change |
| Stock up/down in 20 days | 0.53 (logreg) | **0.56** (`+macro` logreg, all 6 periods), 0.55 (`+events`) | Lead, not a signal (see below) |
| Stock beat the other stocks | 0.51 | 0.51–0.52 | No change |
| Stock big move | 0.58 | **0.60** (`+events`, all 6 periods) | Better: earnings dates predict volatility |

Rules on the new data alone, no ML:
- **High funding → weaker returns** (crowded longs): ROC-AUC 0.45–0.51. No signal.
- **Post-earnings drift** (buy after a beat): 0.48–0.50. No signal, which matches published research that the effect is weak in mega-cap stocks.

Why the stock 20-day result is a lead and not a signal:
- Only logistic regression finds it. Gradient boosting and random forest on the same features score 0.48–0.50.
- 10 large US stocks and ETFs mostly move together, so the 6 test periods hold only about 6 independent bets on "will the market rise over the next month".
- Its top-vs-bottom spread is about +1% per 20 days, small next to the noise.
- It was found among 172 experiments.

Testing it properly needs a much bigger stock universe (for example 100+ S&P 500 stocks) so the market-direction effect can be separated from stock picking.

The 2023+ crypto window shows 20-day direction at 0.63 with `+cross` alone. That window is one strong crypto bull market, so this most likely reflects "trend kept going" in that regime. Adding funding made it worse (0.58).

## Round 3: large universe (99 stocks, 39 coins)

`python -m src.research --universe large --quick`. Symbols are listed in `config.toml`. COMP-USD had no Yahoo data. Per-coin funding is left out because Hyperliquid doesn't list most of these coins. Full tables: `large/signal_research.md`.

| Question | 10 symbols | Large universe | Verdict |
|---|---|---|---|
| Stock up/down in 20 days, `+macro` (the round-2 lead) | 0.56, all 6 periods | 0.52, 3 of 6 | **Lead gone.** It was a market-timing fluke on 10 correlated names |
| Stock up/down in 5 days | 0.51 | 0.51 | Not predictable |
| Stock beat the other stocks | 0.51 | 0.51 | No stock-picking signal |
| Stock big move | 0.58–0.60 | 0.55–0.56, all 6 periods | Predictable |
| Crypto beat the other coins | 0.53 | 0.53, all 6 periods | Weak but consistent |
| Crypto big move | 0.55 | 0.56, all 6 periods | Predictable |
| Crypto up/down in 20 days | 0.53–0.55 | 0.58, all 6 periods (logreg) | Lead, same caveat as before: coins move together |

### Rotation on the large universe

Hold the top-k symbols by predicted "beats the median", every 5–20 days, with costs. 13 configurations per asset class, all reported.

**Stocks (99):** ML beat the equal-weight basket in 2 of 13 configurations and momentum in 1 of 13. No edge.

**Crypto (39), June 2021 to Nov 2026:**
- ML beat the equal-weight basket of all 39 coins in **13 of 13** configurations. The basket lost about 45%; ML returns ranged from −35% to +621%.
- ML beat random picks in 98–100% of runs.
- ML beat a plain 20-day momentum rule in 8 of 13.
- Features with BTC/market context (`+cross`, `all`) did most of the work. Price-only features (`base`) were far weaker.

**Why this is a lead, not a strategy:**
- **The coin list was picked in 2026.** Coins that collapsed completely (e.g. LUNA, FTT) aren't in it, and a few of today's winners (ZEC, TRX) are. Most of the gap to the basket is "avoid the coins that bled out", which is easy to over-credit with hindsight. A fair test needs a point-in-time list: the top 40 coins by market cap as of each date.
- **Drawdowns of 71–90%** on every ML configuration. Sharpe ratio 0.4–0.8.
- **Yahoo prices for small coins** carry no liquidity or spread information. Real fills on small coins would cost more than 0.15–0.30%.
- **Momentum gets most of the same result,** so the ML adds a modest amount on top of a known effect.

Data check: no one-day moves over ±50% in the test window look like bad prices. They match real events, such as XRP on 2023-07-13. The extreme early values (SHIB and AAVE in 2020–21) fall in training data only.

## Caveat on multiple testing

Testing 172 combinations means a few will look good by chance. That's why results are judged by consistency across all 6 test periods and across models, not by the single best score. The 20-day crypto result (best 0.552, 5 of 6 periods) is exactly the kind of lone good number that should not be trusted without new data.
