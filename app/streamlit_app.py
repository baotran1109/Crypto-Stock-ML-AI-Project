"""
streamlit_app.py
Interactive dashboard for prediction and visualization

"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.data_loader import load_crypto
from src.features import add_indicators
from src.backtest import run_backtest

st.set_page_config(page_title="Crypto/Stock Machine Learning Predictor", page_icon="🪙", layout="wide")
st.title("Crypto/Stock Machine Learning Predictor 🪙")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Model Performance", "Backtest"])

# Load model and metadata (cache for performance)
@st.cache_data
def load_model_artifacts():
    try:
        model = joblib.load("artifacts/general_crypto_model.pkl")
        cols = joblib.load("artifacts/feature_columns.pkl")
        eval_metrics = None
        feature_importance = None
        
        try:
            eval_metrics = joblib.load("artifacts/evaluation_metrics.pkl")
        except FileNotFoundError:
            pass
        
        try:
            feature_importance = joblib.load("artifacts/feature_importance.pkl")
        except FileNotFoundError:
            pass
        
        return model, cols, eval_metrics, feature_importance
    except FileNotFoundError as e:
        return None, None, None, None

model, cols, eval_metrics, feature_importance = load_model_artifacts()

if model is None:
    st.error("❌ Model files not found. Run `python -m src.train_model` first.")
    st.stop()

# ==========================================================
# PREDICTION PAGE
# ==========================================================
if page == "Prediction":
    symbol = st.text_input("Enter symbol (e.g BTC-USD,ETH-USD,AAPL):", "BTC-USD")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        show_candlestick = st.checkbox("Show Candlestick Chart", value=True)
        chart_days = st.slider("Chart Days", 30, 500, 300)
    
    if st.button("Predict", type="primary"):
        with st.spinner(f"Fetching {symbol} data..."):
            df = load_crypto(symbol, force_refresh=False)
            df = add_indicators(df)
            
            if df.empty:
                st.error("No indicator rows available for this symbol (try a longer lookback or another symbol).")
            else:
                latest = df.tail(1)
                
                # Build X with EXACT training columns (fill missing with 0)
                X = pd.DataFrame(index=latest.index)
                for c in cols:
                    X[c] = latest[c] if c in latest.columns else 0
                X = X[cols]
                
                if X.shape[0] == 0:
                    st.error("Couldn't build a feature row (latest bar may be incomplete). Try again later or use a different symbol.")
                else:
                    pred = model.predict(X)[0]
                    prob = model.predict_proba(X)[0, 1] if hasattr(model, "predict_proba") else 0.5
                    
                    # Display prediction
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prediction", "📈 UP" if pred == 1 else "📉 DOWN")
                    with col2:
                        st.metric("Confidence", f"{prob:.2%}")
                    with col3:
                        current_price = df.iloc[-1]['close']
                        st.metric("Current Price", f"${current_price:,.2f}")
                    
                    # Chart
                    chart_data = df.tail(chart_days)
                    
                    if show_candlestick and all(col in chart_data.columns for col in ['open', 'high', 'low', 'close']):
                        # Candlestick chart
                        fig = make_subplots(
                            rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=(f'{symbol} Price', 'Volume'),
                            row_width=[0.7, 0.3]
                        )
                        
                        # Candlestick
                        fig.add_trace(
                            go.Candlestick(
                                x=chart_data.index,
                                open=chart_data['open'],
                                high=chart_data['high'],
                                low=chart_data['low'],
                                close=chart_data['close'],
                                name="Price"
                            ),
                            row=1, col=1
                        )
                        
                        # Volume
                        colors = ['red' if chart_data['close'].iloc[i] < chart_data['open'].iloc[i] 
                                 else 'green' for i in range(len(chart_data))]
                        fig.add_trace(
                            go.Bar(x=chart_data.index, y=chart_data['volume'], name="Volume", marker_color=colors),
                            row=2, col=1
                        )
                        
                        fig.update_layout(
                            height=600,
                            xaxis_rangeslider_visible=False,
                            showlegend=False,
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Simple line chart
                        st.line_chart(chart_data[["close"]])
                    
                    # Show recent indicators
                    with st.expander("📊 Recent Technical Indicators"):
                        indicator_cols = ['rsi', 'ema12', 'ema26', 'macd', 'bb_low', 'bb_mid', 'bb_high']
                        available_indicators = [col for col in indicator_cols if col in df.columns]
                        if available_indicators:
                            recent_indicators = df[available_indicators].tail(10)
                            st.dataframe(recent_indicators.style.format("{:.2f}"))

# ==========================================================
# MODEL PERFORMANCE PAGE
# ==========================================================
elif page == "Model Performance":
    st.header("📊 Model Performance Metrics")
    
    if eval_metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{eval_metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{eval_metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{eval_metrics['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{eval_metrics['f1_score']:.3f}")
        
        if eval_metrics.get('roc_auc'):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ROC-AUC", f"{eval_metrics['roc_auc']:.3f}")
            with col2:
                baseline = eval_metrics.get('baseline_accuracy', 0.5)
                improvement = eval_metrics['accuracy'] - baseline
                st.metric("vs Baseline", f"+{improvement:.3f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = np.array(eval_metrics['confusion_matrix'])
        cm_df = pd.DataFrame(
            cm,
            index=['Actual DOWN', 'Actual UP'],
            columns=['Predicted DOWN', 'Predicted UP']
        )
        st.dataframe(cm_df, use_container_width=True)
        
        # Feature Importance
        if feature_importance:
            st.subheader("🔍 Feature Importance")
            feature_df = pd.DataFrame(
                list(feature_importance.items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)
            
            # Bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=feature_df['Importance'],
                    y=feature_df['Feature'],
                    orientation='h',
                    marker=dict(color=feature_df['Importance'], colorscale='Viridis')
                )
            ])
            fig.update_layout(
                title="Top Features by Importance",
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=600,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            with st.expander("View All Features"):
                st.dataframe(feature_df, use_container_width=True)
    else:
        st.warning("⚠️ Evaluation metrics not found. Run `python -m src.train_model` to generate metrics.")

# ==========================================================
# BACKTEST PAGE
# ==========================================================
elif page == "Backtest":
    st.header("🧪 Strategy Backtesting")
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Symbol to Backtest", "BTC-USD", 
                              help="Use format: BTC-USD, ETH-USD, AAPL, TSLA, etc. (Not BTC-USDT)")
    with col2:
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Show common symbols as suggestions
    with st.expander("💡 Common Symbols"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Cryptocurrencies:**")
            st.write("BTC-USD, ETH-USD, SOL-USD")
            st.write("XRP-USD, MATIC-USD, DOGE-USD")
        with col2:
            st.write("**Stocks:**")
            st.write("AAPL, TSLA, MSFT, GOOGL")
            st.write("META, NVDA, AMZN, SPY")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        initial_capital = st.number_input("Initial Capital ($)", 1000, 1000000, 10000)
    with col2:
        commission = st.number_input("Commission (%)", 0.0, 1.0, 0.1, 0.01) / 100
    with col3:
        lookback_years = st.selectbox("Lookback Period", ["1y", "2y", "3y", "5y"], index=3)
    
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest... This may take a minute."):
            results = run_backtest(
                symbol=symbol,
                confidence_threshold=confidence_threshold,
                initial_capital=initial_capital,
                commission=commission,
                lookback=lookback_years
            )
            
            if "error" in results:
                st.error(f"❌ {results['error']}")
                
                # Show diagnostics if available
                if "diagnostics" in results:
                    with st.expander("🔍 Diagnostic Information"):
                        diag = results['diagnostics']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Buy Signals", diag.get('buy_signals', 0))
                            st.metric("Sell Signals", diag.get('sell_signals', 0))
                        with col2:
                            st.metric("Neutral Signals", diag.get('neutral_signals', 0))
                            st.metric("Data Points", diag.get('data_points_available', 0))
                        with col3:
                            st.metric("Total Predictions", diag.get('total_predictions_processed', 0))
                            st.metric("Confidence Threshold", f"{diag.get('confidence_threshold', 0.5):.2f}")
                
                # Provide additional help
                if "USDT" in symbol.upper() or "not found" in results['error'].lower():
                    st.info("💡 **Tip:** Use '-USD' format for cryptocurrencies (e.g., BTC-USD, not BTC-USDT). For stocks, use the ticker symbol (e.g., AAPL, TSLA).")
                elif "No data after adding indicators" in results['error']:
                    st.info("💡 **Tip:** Try using a longer lookback period (5y) or a different symbol.")
                elif "No trades executed" in results['error']:
                    if "suggestion" in results:
                        st.warning(f"💡 **Suggestion:** {results['suggestion']}")
                    else:
                        st.info("💡 **Tip:** Try lowering the confidence threshold to 0.30-0.40, or use a longer lookback period (5y) to get more trading opportunities.")
            else:
                # Key Metrics
                st.subheader("📈 Backtest Results")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", f"{results['total_return']:.2%}")
                    st.metric("Buy & Hold Return", f"{results['buy_hold_return']:.2%}")
                with col2:
                    st.metric("Final Capital", f"${results['final_capital']:,.2f}")
                    st.metric("Excess Return", f"{results['excess_return']:.2%}")
                with col3:
                    st.metric("Win Rate", f"{results['win_rate']:.2%}")
                    st.metric("Total Trades", f"{results['total_trades']}")
                with col4:
                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                    st.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
                
                # Equity Curve
                if results.get('equity_curve') and len(results['equity_curve']) > 1:
                    st.subheader("💰 Equity Curve")
                    equity_df = pd.DataFrame({
                        'Equity': results['equity_curve'],
                        'Date': pd.to_datetime(results['dates'][:len(results['equity_curve'])])
                    })
                    equity_df = equity_df.set_index('Date')
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_df.index,
                        y=equity_df['Equity'],
                        mode='lines',
                        name='Strategy Equity',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add buy & hold comparison
                    if len(equity_df) > 0:
                        buy_hold_start = equity_df['Equity'].iloc[0]
                        buy_hold_end = buy_hold_start * (1 + results['buy_hold_return'])
                        buy_hold_line = np.linspace(buy_hold_start, buy_hold_end, len(equity_df))
                        fig.add_trace(go.Scatter(
                            x=equity_df.index,
                            y=buy_hold_line,
                            mode='lines',
                            name='Buy & Hold',
                            line=dict(color='gray', width=2, dash='dash')
                        ))
                    
                    fig.update_layout(
                        title="Equity Curve vs Buy & Hold",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        height=400,
                        template="plotly_white",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Trade Details
                if results.get('trades') and len(results['trades']) > 0:
                    st.subheader("📋 Trade History")
                    trades_df = pd.DataFrame(results['trades'])
                    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
                    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
                    trades_df['return'] = trades_df['return'] * 100
                    
                    # Format for display
                    display_cols = ['entry_date', 'exit_date', 'entry_price', 'exit_price', 
                                  'return', 'correct', 'confidence']
                    available_cols = [col for col in display_cols if col in trades_df.columns]
                    st.dataframe(
                        trades_df[available_cols].style.format({
                            'entry_price': '${:.2f}',
                            'exit_price': '${:.2f}',
                            'return': '{:.2f}%',
                            'confidence': '{:.2%}'
                        }),
                        use_container_width=True,
                        height=300
                    )
                    
                    # Trade Statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Win", f"{results['avg_win']:.2%}")
                        st.metric("Average Loss", f"{results['avg_loss']:.2%}")
                    with col2:
                        st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
                        st.metric("Prediction Accuracy", f"{results['prediction_accuracy']:.2%}")

