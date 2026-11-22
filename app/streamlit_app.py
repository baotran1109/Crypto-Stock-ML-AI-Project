"""
streamlit_app.py
Interactive dashboard for prediction and visualization

"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.data_loader import load_crypto
from src.features import add_indicators
from src.backtest import run_backtest
from src.model_registry import ASSET_CLASS_LABELS, TRAINING_UNIVERSES, asset_class_for, load_artifacts

st.set_page_config(page_title="Crypto/Stock Machine Learning Predictor", page_icon="🪙", layout="wide")
st.title("Crypto/Stock Machine Learning Predictor 🪙")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Model Performance", "Backtest", "Research"])

# One model per asset class: crypto and stocks are trained separately (cached for performance)
@st.cache_resource
def load_all_models():
    return {ac: load_artifacts(ac) for ac in TRAINING_UNIVERSES}

bundles = load_all_models()

if not any(bundles.values()):
    st.error("❌ Model files not found. Run `python -m src.train_model` first.")
    st.stop()

st.sidebar.markdown("**Models**")
for ac, b in bundles.items():
    label = ASSET_CLASS_LABELS[ac]
    if b is None:
        st.sidebar.caption(f"{label}: not trained. Run `python -m src.train_model --classes {ac}`.")
    elif b.get("metadata"):
        m = b["metadata"]
        st.sidebar.caption(
            f"**{label}** model: {', '.join(m['symbols'])}\n\n"
            f"Train {m['train_start']} → {m['train_end']} · Test {m['test_start']} → {m['test_end']}"
        )

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
        asset_class = asset_class_for(symbol)
        bundle = bundles.get(asset_class)
        if bundle is None:
            st.error(f"No {ASSET_CLASS_LABELS[asset_class].lower()} model trained. "
                     f"Run `python -m src.train_model --classes {asset_class}`.")
            st.stop()
        model, cols, training_meta = bundle["model"], bundle["feature_cols"], bundle["metadata"]
        with st.spinner(f"Fetching {symbol} data..."):
            try:
                df = add_indicators(load_crypto(symbol, force_refresh=False))
            except Exception as e:
                df = None
                st.error(f"Could not load data for '{symbol}': {e}")

            if df is None:
                pass
            elif df.empty:
                st.error("No indicator rows available for this symbol (try a longer lookback or another symbol).")
            else:
                # Same features the model was trained on, from the latest complete bar
                missing = [c for c in cols if c not in df.columns]
                X = df[cols].dropna().tail(1) if not missing else pd.DataFrame()

                if missing:
                    st.error(f"Saved model expects features {missing}. Retrain with `python -m src.train_model`.")
                elif X.empty:
                    st.error("Couldn't build a feature row (latest bar may be incomplete). Try again later or use a different symbol.")
                else:
                    prob = model.predict_proba(X)[0, 1]
                    pred = int(prob > 0.5)
                    horizon = training_meta.get("horizon", 5) if training_meta else 5
                    st.caption(f"Using the **{ASSET_CLASS_LABELS[asset_class].lower()} model**. "
                               f"Prediction for {horizon} bars after the {X.index[-1].date()} close.")
                    if training_meta and symbol.upper() not in [s.upper() for s in training_meta["symbols"]]:
                        st.info(f"ℹ️ {symbol.upper()} was not one of the symbols this model was trained on "
                                f"({', '.join(training_meta['symbols'])}). It is the same asset class, so the model "
                                "applies, but it has never seen this symbol's own history.")
                    
                    # Display prediction
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prediction", "📈 UP" if pred == 1 else "📉 DOWN")
                    with col2:
                        st.metric("P(up)", f"{prob:.2%}")
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
                        st.plotly_chart(fig)
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
    available = [ac for ac, b in bundles.items() if b is not None]
    perf_class = st.radio("Model", available, format_func=lambda ac: ASSET_CLASS_LABELS[ac], horizontal=True)
    bundle = bundles[perf_class]
    eval_metrics, feature_importance, training_meta = bundle["metrics"], bundle["importance"], bundle["metadata"]
    if training_meta:
        st.caption(f"Trained on {', '.join(training_meta['symbols'])}. "
                   f"Target: is the close {training_meta.get('horizon', 5)} bars ahead higher than today's close?")

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
                st.metric("Accuracy vs Baseline", f"{improvement:+.3f}",
                          help=f"Baseline = always predicting the majority class ({baseline:.3f})")
            st.caption("ROC-AUC of 0.5 means no better than random. Metrics are on the held-out test period"
                       + (f" ({training_meta['test_start']} → {training_meta['test_end']})." if training_meta else "."))

        if eval_metrics.get('cv_accuracy') is not None:
            col1, col2 = st.columns(2)
            col1.metric("Walk-forward CV Accuracy", f"{eval_metrics['cv_accuracy']:.3f}")
            col2.metric("Walk-forward CV ROC-AUC", f"{eval_metrics['cv_roc_auc']:.3f}")

        if eval_metrics.get('per_symbol'):
            st.subheader("Per-symbol test results")
            st.dataframe(pd.DataFrame(eval_metrics['per_symbol']).T, width="stretch")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = np.array(eval_metrics['confusion_matrix'])
        cm_df = pd.DataFrame(
            cm,
            index=['Actual DOWN', 'Actual UP'],
            columns=['Predicted DOWN', 'Predicted UP']
        )
        st.dataframe(cm_df, width="stretch")

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
            st.plotly_chart(fig)

            # Table
            with st.expander("View All Features"):
                st.dataframe(feature_df, width="stretch")
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

    out_of_sample = st.checkbox(
        "Out-of-sample only (after the model's training period)", value=True,
        help="Unticking this lets the backtest include dates the model was trained on, which inflates results.")

    with st.expander("⚙️ Risk rules (applied to the ML model and every benchmark)"):
        col1, col2, col3 = st.columns(3)
        with col1:
            use_vol_target = st.checkbox("Volatility-based sizing", value=False,
                                         help="Invest less when the market is volatile: position = target vol / current vol, capped at 100%.")
            vol_target_pct = st.slider("Target volatility (annual %)", 5, 100, 40, 5, disabled=not use_vol_target)
        with col2:
            min_hold = st.number_input("Minimum holding period (bars)", 0, 60, 0, 1,
                                       help="Ignore exit signals until a position has been held this long. Cuts back-and-forth trading.")
        with col3:
            slippage_pct = st.number_input("Slippage per side (%)", 0.0, 1.0, 0.05, 0.01,
                                           help="Extra cost per trade on top of commission, for fills worse than the close.")
    
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest... This may take a minute."):
            results = run_backtest(
                symbol=symbol,
                confidence_threshold=confidence_threshold,
                initial_capital=initial_capital,
                commission=commission,
                lookback=lookback_years,
                out_of_sample=out_of_sample,
                vol_target=vol_target_pct / 100 if use_vol_target else None,
                min_hold=int(min_hold),
                slippage=slippage_pct / 100,
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
                st.caption(f"{ASSET_CLASS_LABELS[results['asset_class']]} model · "
                           f"Period: {results['period_start']} → {results['period_end']} · "
                           f"buy if P(up) > {results['buy_threshold']:.3f}, exit if P(up) < {results['sell_threshold']:.3f}")
                if results.get('in_sample_overlap'):
                    st.warning("⚠️ This period overlaps the data the model was trained on for this symbol, "
                               "so these results are optimistic.")
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

                    # Buy & hold: the actual price path, scaled to the starting capital
                    if results.get('buy_hold_curve'):
                        fig.add_trace(go.Scatter(
                            x=equity_df.index,
                            y=results['buy_hold_curve'][:len(equity_df)],
                            mode='lines',
                            name='Buy & Hold',
                            line=dict(color='gray', width=2, dash='dash')
                        ))

                    # Other rule-based benchmarks (click legend entries to toggle)
                    bench_colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
                    others = [b for b in results.get('benchmarks', []) if b['name'] != 'Buy & hold']
                    for b, color in zip(others, bench_colors):
                        fig.add_trace(go.Scatter(
                            x=equity_df.index,
                            y=b['equity_curve'][:len(equity_df)],
                            mode='lines',
                            name=b['name'],
                            line=dict(color=color, width=1.5, dash='dot')
                        ))

                    fig.update_layout(
                        title="Equity Curve vs Benchmarks",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        height=400,
                        template="plotly_white",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig)

                # Benchmark comparison
                if results.get('benchmarks'):
                    st.subheader("🏁 ML Model vs Benchmarks")
                    st.caption("Every strategy runs through the same engine, window, capital and commission. "
                               "If the ML model can't beat these simple rules, it isn't adding value.")
                    rows = [results['ml_summary']] + results['benchmarks']
                    comp = pd.DataFrame([{
                        'Strategy': r['name'],
                        'Total Return': r['total_return'],
                        'Sharpe': r['sharpe_ratio'],
                        'Max Drawdown': r['max_drawdown'],
                        'Trades': r['total_trades'],
                        'Win Rate': r['win_rate'],
                        'Time in Market': r['time_in_market'],
                        'Avg Exposure': r.get('avg_exposure', r['time_in_market']),
                        'Rule': r.get('rule', 'Buy/sell on model confidence thresholds.'),
                    } for r in rows]).set_index('Strategy')
                    st.dataframe(
                        comp.style.format({
                            'Total Return': '{:+.2%}', 'Sharpe': '{:+.2f}', 'Max Drawdown': '{:.2%}',
                            'Win Rate': '{:.0%}', 'Time in Market': '{:.0%}', 'Avg Exposure': '{:.0%}',
                        }).highlight_max(subset=['Total Return', 'Sharpe'], color='#d4edda'),
                        width="stretch",
                    )
                    beaten = [b['name'] for b in results['benchmarks']
                              if results['ml_summary']['total_return'] > b['total_return']]
                    st.write(f"ML model beat **{len(beaten)} of {len(results['benchmarks'])}** benchmarks on total return"
                             + (f" ({', '.join(beaten)})." if beaten else "."))

                shuffle = results.get('shuffle_test')
                if shuffle:
                    st.subheader("🎲 Shuffle Test")
                    st.caption(f"The ML model's own buy/sell/hold signals were randomly reordered {shuffle['n']} times: "
                               "same number of trades, random timing. A model with real skill should beat most of them.")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("ML Model Return", f"{results['total_return']:+.2%}")
                    col2.metric("Median Shuffled Return", f"{shuffle['median_return']:+.2%}",
                                help=f"10th-90th percentile: {shuffle['p10_return']:+.2%} to {shuffle['p90_return']:+.2%}")
                    col3.metric("ML Beats Shuffles", f"{shuffle['ml_beats_pct']:.0%}")
                    if shuffle['ml_beats_pct'] >= 0.9 and results.get('in_sample_overlap'):
                        st.warning("The model's timing beats random timing here, but this window overlaps the "
                                   "training data, so that is expected and not evidence of skill.")
                    elif shuffle['ml_beats_pct'] >= 0.9:
                        st.success("The model's timing beats random timing in this window.")
                    else:
                        st.info("The model's timing is not clearly better than random timing in this window.")

                # Trade Details
                if results.get('trades') and len(results['trades']) > 0:
                    st.subheader("📋 Trade History")
                    trades_df = pd.DataFrame(results['trades'])
                    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
                    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
                    trades_df['return'] = trades_df['return'] * 100

                    # Format for display
                    display_cols = ['entry_date', 'exit_date', 'entry_price', 'exit_price',
                                  'return', 'correct', 'confidence', 'open_at_end']
                    available_cols = [col for col in display_cols if col in trades_df.columns]
                    st.dataframe(
                        trades_df[available_cols].style.format({
                            'entry_price': '${:.2f}',
                            'exit_price': '${:.2f}',
                            'return': '{:.2f}%',
                            'confidence': '{:.2%}'
                        }),
                        width="stretch",
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

# ==========================================================
# RESEARCH PAGE
# ==========================================================
elif page == "Research":
    st.header("🔬 Signal Research")
    st.caption("Results of `python -m src.research`: many target / feature / model combinations scored "
               "walk-forward on data each model never saw. Re-run that command to refresh this page.")
    for name, label in [("FINDINGS.md", "Findings"), ("signal_research.md", "Full results")]:
        path = os.path.join("reports", name)
        if os.path.exists(path):
            with st.expander(label, expanded=(name == "FINDINGS.md")):
                with open(path, encoding="utf-8") as fh:
                    st.markdown(fh.read())
    if not os.path.exists(os.path.join("reports", "signal_research.md")):
        st.info("No research results yet. Run `python -m src.research` from the project root.")
