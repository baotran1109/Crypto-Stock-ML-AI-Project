"""
backtest.py
Backtesting framework using backtrader to validate if predictions would have made money.
"""

import pandas as pd
import numpy as np
import joblib
import backtrader as bt
from typing import Dict, Optional
from src.data_loader import load_crypto, DataLoadError
from src.features import add_indicators


class MLStrategy(bt.Strategy):
    """Backtrader strategy that uses ML model predictions for trading signals."""
    
    params = (
        ('confidence_threshold', 0.5),
        ('prediction_horizon', 5),
        ('min_lookback', 60),
    )
    
    def __init__(self):
        # Load ML model and features
        self.model = joblib.load("artifacts/general_crypto_model.pkl")
        self.feature_cols = joblib.load("artifacts/feature_columns.pkl")
        
        # Store predictions and trades
        self.predictions = []
        self.probabilities = []
        self.trades_list = []
        self.buy_signals = 0
        self.sell_signals = 0
        self.neutral_signals = 0
        self.skipped_predictions = 0
        
        # Calculate threshold delta (same logic as before)
        threshold_delta = 0.01 + (self.params.confidence_threshold * 0.09)
        self.buy_threshold = 0.5 + threshold_delta
        self.sell_threshold = 0.5 - (threshold_delta * 0.7)  # Asymmetric
        
        # Track current position entry
        self.entry_price = None
        self.entry_date = None
        
        # Store equity curve
        self.equity_curve = []
        self.equity_dates = []
        
    def next(self):
        # Need enough history for indicators
        if len(self.data) < self.params.min_lookback:
            return
        
        # Get current bar index in the original dataframe
        current_idx = len(self.data) - 1
        
        # Access the dataframe with features (stored as class variable)
        if not hasattr(MLStrategy, 'df_with_features'):
            return
            
        if current_idx >= len(MLStrategy.df_with_features):
            return
        
        # Get current row with all indicators
        current_row = MLStrategy.df_with_features.iloc[current_idx:current_idx+1]
        
        # Build feature vector
        X = pd.DataFrame(index=current_row.index)
        for col in self.feature_cols:
            X[col] = current_row[col] if col in current_row.columns else 0
        X = X[self.feature_cols]
        
        if X.isnull().any().any():
            self.neutral_signals += 1
            self.skipped_predictions += 1
            # Update equity curve
            self._update_equity()
            return
        
        # Make prediction
        if hasattr(self.model, 'predict_proba'):
            prob = self.model.predict_proba(X)[0, 1]
            self.probabilities.append(prob)
            
            # Determine signal
            if prob > self.buy_threshold:
                signal = 1  # Buy
                self.buy_signals += 1
            elif prob < self.sell_threshold:
                signal = 0  # Sell
                self.sell_signals += 1
            else:
                signal = -1  # Neutral
                self.neutral_signals += 1
        else:
            signal = self.model.predict(X)[0]
            prob = 0.5
            self.probabilities.append(prob)
            if signal == 1:
                self.buy_signals += 1
            elif signal == 0:
                self.sell_signals += 1
            else:
                self.neutral_signals += 1
        
        self.predictions.append(signal)
        
        # Trading logic
        if signal == 1 and not self.position:  # Buy signal, no position
            # Calculate size based on available cash
            cash = self.broker.getcash()
            price = self.data.close[0]
            if cash > 0 and price > 0:
                size = int(cash / price)
                if size > 0:
                    self.buy(size=size)
        elif signal == 0 and self.position:  # Sell signal, have position
            # Close the entire position
            self.close()
        
        # Update equity curve
        self._update_equity()
    
    def _update_equity(self):
        """Update equity curve at each bar."""
        current_value = self.broker.getvalue()
        self.equity_curve.append(current_value)
        self.equity_dates.append(self.data.datetime.date(0))
    
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                # Store entry info when buy order completes
                self.entry_price = order.executed.price
                self.entry_date = self.data.datetime.date(0)
            elif order.issell():
                # Record completed trade when sell order completes
                if self.entry_price is not None and self.entry_date is not None:
                    trade_return = (order.executed.price - self.entry_price) / self.entry_price
                    
                    # Get actual direction (5 days ahead from entry)
                    entry_idx = None
                    try:
                        if isinstance(self.entry_date, pd.Timestamp):
                            entry_idx = MLStrategy.df_with_features.index.get_loc(self.entry_date)
                        else:
                            # Try to find by date
                            entry_idx = MLStrategy.df_with_features.index.get_indexer([pd.Timestamp(self.entry_date)])[0]
                            if entry_idx < 0:
                                entry_idx = None
                    except (KeyError, IndexError, TypeError):
                        entry_idx = None
                    
                    actual_direction = 1
                    if entry_idx is not None and entry_idx + 5 < len(MLStrategy.df_with_features):
                        future_price = MLStrategy.df_with_features.iloc[entry_idx + 5]['close']
                        actual_direction = 1 if future_price > self.entry_price else 0
                    else:
                        actual_direction = 1 if order.executed.price > self.entry_price else 0
                    
                    # Get confidence from when we entered (find closest prediction)
                    confidence = 0.5
                    if self.probabilities:
                        # Use the last probability before this sell
                        confidence = self.probabilities[-1] if self.probabilities else 0.5
                    
                    self.trades_list.append({
                        'entry_date': self.entry_date,
                        'exit_date': self.data.datetime.date(0),
                        'entry_price': self.entry_price,
                        'exit_price': order.executed.price,
                        'direction': 'LONG',
                        'prediction': 'UP',
                        'actual_direction': actual_direction,
                        'return': trade_return,
                        'correct': 1 if actual_direction == 1 else 0,
                        'confidence': confidence
                    })
                    self.entry_price = None
                    self.entry_date = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            pass


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
) -> Dict:
    """
    Run backtest using backtrader framework.
    
    Returns same format as before for Streamlit compatibility.
    """
    # Load and prepare data
    try:
        df = load_crypto(symbol, lookback=lookback, force_refresh=False)
    except DataLoadError as e:
        if "USDT" in symbol.upper():
            suggested = symbol.replace('-USDT', '-USD').replace('-usdt', '-USD')
            return {"error": f"Symbol '{symbol}' not found. Try '{suggested}' instead (e.g., BTC-USD, ETH-USD)"}
        else:
            return {"error": f"Could not load data for '{symbol}'. Try: BTC-USD, ETH-USD, SOL-USD, AAPL, TSLA, or other valid symbols."}
    except Exception as e:
        return {"error": f"Error loading data for '{symbol}': {str(e)}"}
    
    if df.empty:
        return {"error": f"No data available for symbol '{symbol}'. Try a different symbol or longer lookback period."}
    
    # Filter by date range if provided
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]
    
    if len(df) < 100:
        return {"error": "Not enough data for backtesting"}
    
    # Add indicators
    df = add_indicators(df)
    if df.empty:
        return {"error": "No data after adding indicators"}
    
    # Store dataframe with features as class variable for strategy access
    MLStrategy.df_with_features = df
    
    # Prepare data for backtrader (needs OHLCV format with proper column names)
    df_bt = df[['open', 'high', 'low', 'close', 'volume']].copy()
    df_bt.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_bt.index.name = 'Date'
    
    # Create backtrader engine
    cerebro = bt.Cerebro()
    
    # Add data feed
    data_feed = bt.feeds.PandasData(
        dataname=df_bt,
        datetime=None,
        open=0,
        high=1,
        low=2,
        close=3,
        volume=4,
        openinterest=-1
    )
    cerebro.adddata(data_feed)
    
    # Add strategy with ML model
    cerebro.addstrategy(
        MLStrategy,
        confidence_threshold=confidence_threshold,
        prediction_horizon=5,
        min_lookback=60
    )
    
    # Set initial capital and commission
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=commission)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Run backtest
    try:
        results = cerebro.run()
        strat = results[0]
    except Exception as e:
        return {"error": f"Backtest execution failed: {str(e)}"}
    
    # Extract results
    initial_value = cerebro.broker.startingcash
    final_value = cerebro.broker.getvalue()
    
    # Get analyzer results
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    returns_analysis = strat.analyzers.returns.get_analysis()
    
    # Process trades
    trades_df = pd.DataFrame(strat.trades_list) if strat.trades_list else pd.DataFrame()
    
    # Check if we have trades (after processing open positions)
    if len(trades_df) == 0 and not strat.position:
        total_predictions = strat.buy_signals + strat.sell_signals + strat.neutral_signals + strat.skipped_predictions
        return {
            "error": "No trades executed",
            "symbol": symbol,
            "initial_capital": initial_capital,
            "final_capital": final_value,
            "diagnostics": {
                "buy_signals": strat.buy_signals,
                "sell_signals": strat.sell_signals,
                "neutral_signals": strat.neutral_signals,
                "total_predictions_processed": total_predictions,
                "confidence_threshold": confidence_threshold,
                "data_points_available": len(df) - 60 - 5
            },
            "suggestion": "Try lowering the confidence threshold or using a longer lookback period."
        }
    
    # Close any open position at the end
    if strat.position:
        final_price = df.iloc[-1]['close']
        if strat.entry_price is not None:
            trade_return = (final_price - strat.entry_price) / strat.entry_price
            entry_idx = df.index.get_loc(strat.entry_date) if isinstance(strat.entry_date, pd.Timestamp) else None
            if entry_idx is None:
                try:
                    entry_idx = df.index.get_indexer([pd.Timestamp(strat.entry_date)])[0]
                except:
                    entry_idx = None
            
            actual_direction = 1
            if entry_idx is not None and entry_idx + 5 < len(df):
                future_price = df.iloc[entry_idx + 5]['close']
                actual_direction = 1 if future_price > strat.entry_price else 0
            else:
                actual_direction = 1 if final_price > strat.entry_price else 0
            
            trades_df = pd.concat([trades_df, pd.DataFrame([{
                'entry_date': strat.entry_date,
                'exit_date': df.index[-1],
                'entry_price': strat.entry_price,
                'exit_price': final_price,
                'direction': 'LONG',
                'prediction': 'UP',
                'actual_direction': actual_direction,
                'return': trade_return,
                'correct': 1 if actual_direction == 1 else 0,
                'confidence': strat.probabilities[-1] if strat.probabilities else 0.5
            }])], ignore_index=True)
    
    # Calculate metrics
    total_return = (final_value - initial_capital) / initial_capital
    min_lookback = 60
    buy_hold_return = (df.iloc[-1]['close'] - df.iloc[min_lookback]['close']) / df.iloc[min_lookback]['close']
    
    winning_trades = trades_df[trades_df['return'] > 0]
    losing_trades = trades_df[trades_df['return'] <= 0]
    
    win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
    avg_win = winning_trades['return'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['return'].mean() if len(losing_trades) > 0 else 0
    profit_factor = abs(winning_trades['return'].sum() / losing_trades['return'].sum()) if len(losing_trades) > 0 and losing_trades['return'].sum() != 0 else 0
    
    prediction_accuracy = trades_df['correct'].mean() if len(trades_df) > 0 else 0
    
    # Get metrics from analyzers
    sharpe_ratio = sharpe_analysis.get('sharperatio', 0)
    if sharpe_ratio is None:
        sharpe_ratio = 0
    
    max_drawdown = abs(drawdown_analysis.get('max', {}).get('drawdown', 0)) / 100 if drawdown_analysis.get('max', {}).get('drawdown') else 0
    
    # Use equity curve from strategy
    equity_curve = strat.equity_curve if strat.equity_curve else [initial_capital] * len(df)
    dates = [str(d) for d in strat.equity_dates] if strat.equity_dates else [str(d) for d in df.index[60:]]
    
    # Ensure same length
    min_len = min(len(equity_curve), len(dates))
    equity_curve = equity_curve[:min_len]
    dates = dates[:min_len]
    
    results = {
        "symbol": symbol,
        "initial_capital": initial_capital,
        "final_capital": final_value,
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "excess_return": total_return - buy_hold_return,
        "total_trades": len(trades_df),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "prediction_accuracy": prediction_accuracy,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio if isinstance(sharpe_ratio, (int, float)) else 0,
        "trades": trades_df.to_dict('records') if len(trades_df) > 0 else [],
        "equity_curve": equity_curve,
        "dates": dates
    }
    
    return results
