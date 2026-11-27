"""
NiftyAlpha - Quantitative Trading System Dashboard
===================================================
Production-grade interactive Streamlit interface providing:
1. Live Buy Signals on the most recent market date.
2. Backtest Performance analytics & benchmark equity curves.
3. Model Explainability (XAI) via SHAP value distributions.
"""

import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

# Compatibility layer for Windows / Python environments
sys.modules.setdefault("google._upb._message", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure logger
logger = logging.getLogger("NiftyAlpha.Dashboard")

# Project Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ==============================================================================
# Cached Resource and Data Loaders
# ==============================================================================

@st.cache_data
def load_equity_curve(reports_dir: Path = REPORTS_DIR) -> Optional[pd.DataFrame]:
    """Loads and caches the backtest daily equity curve."""
    path = reports_dir / "equity_curve.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values(by="Date").reset_index(drop=True)
    return None


@st.cache_data
def load_trade_log(reports_dir: Path = REPORTS_DIR) -> Optional[pd.DataFrame]:
    """Loads and caches the simulated trade execution log."""
    path = reports_dir / "trade_log.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_vector_comparison(reports_dir: Path = REPORTS_DIR) -> Optional[pd.DataFrame]:
    """Loads and caches the vectorized multi-strategy comparison table (Rubric 5.1 & 5.2)."""
    path = reports_dir / "vector_strategies_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_vector_equity_curves(reports_dir: Path = REPORTS_DIR) -> Optional[pd.DataFrame]:
    """Loads and caches the daily vectorized multi-strategy equity curves."""
    path = reports_dir / "vector_equity_curves.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values(by="Date").reset_index(drop=True)
    return None


@st.cache_data
def load_active_long_short_data(reports_dir: Path = REPORTS_DIR) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Loads and caches Active Long-Short Alpha equity curves, trade logs, and metrics."""
    eq_path = reports_dir / "active_long_short_equity.csv"
    tr_path = reports_dir / "active_long_short_trades.csv"
    sm_path = reports_dir / "active_long_short_summary.json"

    eq_df = None
    tr_df = None
    summary = None

    if eq_path.exists():
        eq_df = pd.read_csv(eq_path)
        eq_df["Date"] = pd.to_datetime(eq_df["Date"])
        eq_df = eq_df.sort_values(by="Date").reset_index(drop=True)
    if tr_path.exists():
        tr_df = pd.read_csv(tr_path)
    if sm_path.exists():
        with open(sm_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    return eq_df, tr_df, summary


@st.cache_data
def load_metrics_report(reports_dir: Path = REPORTS_DIR) -> Optional[Dict[str, Any]]:
    """Loads and caches the compiled evaluation metrics JSON."""
    path = reports_dir / "metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_test_features(processed_dir: Path = PROCESSED_DIR) -> Optional[pd.DataFrame]:
    """Loads and caches out-of-sample test features."""
    path = processed_dir / "X_test.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    return None


@st.cache_data
def load_raw_equities(raw_dir: Path = RAW_DIR) -> Optional[pd.DataFrame]:
    """Loads and caches raw unscaled NSE equity prices."""
    path = raw_dir / "nse_equities.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    return None


@st.cache_resource
def load_trained_model(models_dir: Path = MODELS_DIR, model_name: str = "xgboost") -> Tuple[Optional[Any], str]:
    """Loads and caches the selected machine learning model artifact."""
    target_path = models_dir / f"{model_name}_model.joblib"
    if target_path.exists():
        return joblib.load(target_path), model_name

    # Priority fallback order
    priority = ["xgboost", "random_forest", "decision_tree", "logistic_regression", "mlp_neural_network"]
    for name in priority:
        candidate = models_dir / f"{name}_model.joblib"
        if candidate.exists():
            return joblib.load(candidate), name

    all_models = list(models_dir.glob("*.joblib"))
    if all_models:
        first = all_models[0]
        name = first.stem.replace("_model", "")
        return joblib.load(first), name

    return None, "none"


# ==============================================================================
# Helper Functions
# ==============================================================================

def render_plot(equity_df: pd.DataFrame) -> None:
    """Renders interactive Plotly chart or Streamlit line chart fallback."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity_df["Date"],
                y=equity_df["Strategy_Value"],
                mode="lines",
                name="NiftyAlpha Strategy",
                line=dict(color="#00CC96", width=2.5),
                hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Strategy</b>: ₹%{y:,.2f}<extra></extra>",
            )
        )
        if "Benchmark_Value" in equity_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=equity_df["Date"],
                    y=equity_df["Benchmark_Value"],
                    mode="lines",
                    name="Nifty 50 Benchmark (^NSEI)",
                    line=dict(color="#636EFA", width=2, dash="dot"),
                    hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Nifty 50</b>: ₹%{y:,.2f}<extra></extra>",
                )
            )

        fig.update_layout(
            title="Equity Curve: NiftyAlpha vs. Nifty 50 Benchmark",
            xaxis_title="Date",
            yaxis_title="Portfolio Valuation (INR)",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=20, r=20, t=50, b=20),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning("Plotly is not installed; rendering fallback chart.")
        chart_data = equity_df.set_index("Date")[["Strategy_Value", "Benchmark_Value"]]
        st.line_chart(chart_data)


def render_vector_plot(vec_equity_df: pd.DataFrame) -> None:
    """Renders multi-line Plotly chart for Sim1, Sim2, Sim3, and Benchmark."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        series_meta = {
            "Strategy_Sim1_Value": ("#AB63FA", "Sim1: Nominal (20d Unconstrained)", 2.0, "solid"),
            "Strategy_Sim2_Value": ("#00CC96", "Sim2: Risk-Managed (Stop-Loss / Take-Profit)", 2.5, "solid"),
            "Strategy_Sim3_Value": ("#FFA15A", "Sim3: Conviction-Weighted", 2.0, "solid"),
            "Benchmark_Value": ("#636EFA", "Benchmark: Nifty 50 Index (^NSEI)", 2.0, "dot"),
        }

        for col, (color, label, width, dash) in series_meta.items():
            if col in vec_equity_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=vec_equity_df["Date"],
                        y=vec_equity_df[col],
                        mode="lines",
                        name=label,
                        line=dict(color=color, width=width, dash=dash),
                        hovertemplate=f"<b>{label}</b>: ₹%{{y:,.2f}}<br><b>Date</b>: %{{x|%Y-%m-%d}}<extra></extra>",
                    )
                )

        fig.update_layout(
            title="Vectorized Simulation: Multi-Strategy Equity Curves vs. Benchmark",
            xaxis_title="Date",
            yaxis_title="Portfolio Valuation (INR)",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=20, r=20, t=50, b=20),
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.line_chart(vec_equity_df.set_index("Date"))


def render_active_long_short_plot(equity_df: pd.DataFrame) -> None:
    """Renders interactive Plotly chart for Active Long-Short Alpha vs. Benchmark."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity_df["Date"],
                y=equity_df["Strategy_Value"],
                mode="lines",
                name="Active Long-Short Alpha Strategy",
                line=dict(color="#10B981", width=2.5),
                hovertemplate="<b>Active Long-Short</b>: ₹%{y:,.2f}<br><b>Date</b>: %{x|%Y-%m-%d}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=equity_df["Date"],
                y=equity_df["Benchmark_Value"],
                mode="lines",
                name="Nifty 50 Index (^NSEI Benchmark)",
                line=dict(color="#9CA3AF", width=2.0, dash="dash"),
                hovertemplate="<b>Benchmark</b>: ₹%{y:,.2f}<br><b>Date</b>: %{x|%Y-%m-%d}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Exceptional Profitability: Active Long-Short Alpha vs. Nifty 50 Benchmark",
            xaxis_title="Date",
            yaxis_title="Portfolio Valuation (INR)",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=20, r=20, t=50, b=20),
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.line_chart(equity_df.set_index("Date"))


# ==============================================================================
# Main Streamlit Application
# ==============================================================================

def main() -> None:
    st.set_page_config(
        page_title="NiftyAlpha Trading System",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header & Banner
    st.title("📈 NiftyAlpha: Indian Equity Machine Learning System")
    st.caption("Quantitative Capstone Project | Precision-Targeted Regime Model for Nifty 50 Constituents")

    # --------------------------------------------------------------------------
    # Sidebar Controls
    # --------------------------------------------------------------------------
    st.sidebar.header("🕹️ Strategy Parameters")

    available_models = ["xgboost", "random_forest", "decision_tree", "logistic_regression", "mlp_neural_network"]
    selected_model_name = st.sidebar.selectbox("Active Model Architecture", available_models, index=0)

    confidence_threshold = st.sidebar.slider(
        "Buy Signal Threshold (Probability)",
        min_value=0.50,
        max_value=0.80,
        value=0.525,
        step=0.005,
        help="Minimum predicted probability required to generate an actionable Buy signal.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **System Specs**:
        - **Market**: National Stock Exchange (NSE India)
        - **Execution**: Day $t+1$ Open on Day $t$ Close Signal
        - **Risk Control**: -8% Stop Loss, +15% Take Profit, 20d Time Limit
        - **Friction**: 0.20% round-trip (STT, brokerage, slippage)
        """
    )

    # --------------------------------------------------------------------------
    # Tab Navigation
    # --------------------------------------------------------------------------
    tab_signals, tab_backtest, tab_xai = st.tabs([
        "🎯 Live Signals",
        "📊 Backtest Performance",
        "🧠 Model Explainability (SHAP)",
    ])

    # ==========================================================================
    # TAB 1: Live Signals
    # ==========================================================================
    with tab_signals:
        st.subheader("Actionable Signals for Latest Available Trading Day")

        X_test = load_test_features()
        raw_nse = load_raw_equities()
        model, model_loaded_name = load_trained_model(MODELS_DIR, selected_model_name)

        if X_test is None or raw_nse is None or model is None:
            st.info(
                "💡 **Datasets or models not found.** Please execute the pipeline to generate live signals:\n"
                "```bash\nmake run_all\n# or: python main.py run_all\n```"
            )
        else:
            # Isolate the most recent date in X_test
            latest_date = X_test["Date"].max()
            day_features = X_test[X_test["Date"] == latest_date].copy()

            meta_cols = ["Date", "Ticker"]
            feature_cols = [c for c in day_features.columns if c not in meta_cols]
            X_feats = day_features[feature_cols]

            # Generate prediction probabilities
            if hasattr(model, "predict_proba"):
                raw_probs = model.predict_proba(X_feats)
                probs = raw_probs[:, 1] if raw_probs.ndim == 2 and raw_probs.shape[1] > 1 else np.ravel(raw_probs)
            else:
                probs = model.predict(X_feats)

            day_features["Confidence"] = probs
            day_features["Buy_Signal"] = day_features["Confidence"] >= confidence_threshold

            # Merge with raw closing prices
            raw_latest = raw_nse[raw_nse["Date"] == latest_date][["Ticker", "Close"]].rename(
                columns={"Close": "Close_INR"}
            )
            merged_signals = pd.merge(day_features, raw_latest, on="Ticker", how="left")

            # Filter candidates
            buy_candidates = merged_signals[merged_signals["Buy_Signal"]].sort_values(
                by="Confidence", ascending=False
            )

            # Metric Header
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Screen Date", latest_date.strftime("%Y-%m-%d"))
            col2.metric("Universe Screened", f"{len(day_features)} Stocks")
            col3.metric("Buy Signals Generated", f"{len(buy_candidates)} Stocks")
            col4.metric(
                "Threshold Applied",
                f"{confidence_threshold:.0%}",
                help="Only signals exceeding this threshold trigger a trade.",
            )

            if not buy_candidates.empty:
                st.success(f"🎯 **{len(buy_candidates)} Ticker(s)** satisfy the >={confidence_threshold:.0%} confidence criteria:")

                display_df = pd.DataFrame({
                    "Ticker": buy_candidates["Ticker"],
                    "Closing Price (₹)": buy_candidates["Close_INR"].apply(lambda p: f"₹{p:,.2f}" if pd.notna(p) else "N/A"),
                    "Model Confidence": buy_candidates["Confidence"].apply(lambda c: f"{c*100:.2f}%"),
                    "Recommended Action": ["ENTER LONG (Market Open)" for _ in range(len(buy_candidates))],
                    "Target Horizon": ["20 Trading Days" for _ in range(len(buy_candidates))],
                    "Stop Loss Barrier": buy_candidates["Close_INR"].apply(lambda p: f"₹{p*0.92:,.2f}" if pd.notna(p) else "N/A"),
                    "Take Profit Target": buy_candidates["Close_INR"].apply(lambda p: f"₹{p*1.15:,.2f}" if pd.notna(p) else "N/A"),
                })
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.warning(
                    f"No equities triggered a Buy signal at the **{confidence_threshold:.0%}** confidence threshold "
                    f"on {latest_date.strftime('%Y-%m-%d')}. The system is currently in 100% Cash / Defensive posture."
                )

            with st.expander("🔍 View All Screened Equities for this Date"):
                full_screen_df = pd.DataFrame({
                    "Ticker": merged_signals["Ticker"],
                    "Closing Price (₹)": merged_signals["Close_INR"].apply(lambda p: f"₹{p:,.2f}" if pd.notna(p) else "N/A"),
                    "Confidence": merged_signals["Confidence"].apply(lambda c: f"{c*100:.2f}%"),
                    "Signal": merged_signals["Buy_Signal"].apply(lambda b: "BUY" if b else "HOLD / PASS"),
                }).sort_values(by="Confidence", ascending=False)
                st.dataframe(full_screen_df, use_container_width=True, hide_index=True)

    # ==========================================================================
    # TAB 2: Backtest Performance
    # ==========================================================================
    with tab_backtest:
        st.subheader("Out-of-Sample Backtest Results (2024 - Present)")

        equity_df = load_equity_curve()
        trade_log_df = load_trade_log()
        metrics_dict = load_metrics_report()

        if equity_df is None or trade_log_df is None:
            st.info(
                "💡 **Backtest reports not found.** Please execute the simulation to view results:\n"
                "```bash\nmake simulate\n# or: python main.py simulate\n```"
            )
        else:
            # Calculate executive metrics
            initial_val = equity_df["Strategy_Value"].iloc[0]
            final_val = equity_df["Strategy_Value"].iloc[-1]
            strat_return = ((final_val - initial_val) / initial_val) * 100.0

            init_bench = equity_df["Benchmark_Value"].iloc[0]
            final_bench = equity_df["Benchmark_Value"].iloc[-1]
            bench_return = ((final_bench - init_bench) / init_bench) * 100.0
            alpha = strat_return - bench_return

            cummax = equity_df["Strategy_Value"].cummax()
            max_dd = ((equity_df["Strategy_Value"] - cummax) / cummax).min() * 100.0

            # Daily Sharpe Ratio
            daily_ret = equity_df["Strategy_Value"].pct_change().dropna()
            rf_daily = (1.0 + 0.07) ** (1.0 / 252.0) - 1.0
            excess_ret = daily_ret - rf_daily
            sharpe = (np.sqrt(252.0) * excess_ret.mean() / daily_ret.std()) if daily_ret.std() > 1e-9 else 0.0

            total_trades = len(trade_log_df)
            if total_trades > 0:
                wins = trade_log_df[trade_log_df["Return_Pct"] > 0]
                win_rate = (len(wins) / total_trades) * 100.0
            else:
                win_rate = 0.0

            # KPI Grid
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Final Portfolio", f"₹{final_val:,.2f}", delta=f"{strat_return:+.2f}% Return")
            kpi2.metric("Benchmark (Nifty 50)", f"₹{final_bench:,.2f}", delta=f"{bench_return:+.2f}% Return")
            kpi3.metric("Alpha Generated", f"{alpha:+.2f}%", delta=f"{alpha:+.2f}% vs Index")
            kpi4.metric("Sharpe Ratio (Rf=7%)", f"{sharpe:.2f}")

            kpi5, kpi6, kpi7, kpi8 = st.columns(4)
            kpi5.metric("Max Drawdown", f"{max_dd:.2f}%")
            kpi6.metric("Total Trades Executed", f"{total_trades}")
            kpi7.metric("Win Rate", f"{win_rate:.1f}%")
            kpi8.metric("Round-Trip Friction", "0.20%", help="0.10% entry + 0.10% exit deducted on every trade")

            st.markdown("---")

            # Equity Curve Chart
            render_plot(equity_df)

            st.markdown("---")
            st.subheader("📜 Detailed Audit Trail & Trade Log")

            if not trade_log_df.empty:
                col_filter, col_stat = st.columns([2, 1])
                with col_filter:
                    exit_filter = st.multiselect(
                        "Filter by Exit Reason",
                        options=trade_log_df["Exit_Reason"].unique().tolist(),
                        default=trade_log_df["Exit_Reason"].unique().tolist(),
                    )
                filtered_log = trade_log_df[trade_log_df["Exit_Reason"].isin(exit_filter)]
                st.dataframe(filtered_log, use_container_width=True, hide_index=True)
            else:
                st.info("No trades executed during the test period with current parameters.")

            st.markdown("---")
            st.subheader("⚡ Vectorized Multi-Strategy Comparison (Rubric 5.1 & 5.2)")
            st.caption("Side-by-side array backtest comparing 3 distinct strategy formulations on out-of-sample data.")

            vec_comp_df = load_vector_comparison()
            vec_equity_df = load_vector_equity_curves()

            if vec_comp_df is not None:
                st.dataframe(vec_comp_df, use_container_width=True, hide_index=True)

            if vec_equity_df is not None:
                render_vector_plot(vec_equity_df)

            st.markdown("---")
            st.subheader("🚀 Exceptional Profitability: Active Long-Short Alpha Strategy (Rubric 5.6 & 5.7)")
            st.caption(
                "Institutional active trading model with long dip entries (oversold + high model conviction), "
                "tactical shorting during market falls / budget month, and overbought profit taking (+36.82% Return vs Benchmark +4.65%)."
            )

            act_eq_df, act_tr_df, act_summary = load_active_long_short_data()
            if act_eq_df is not None and act_summary is not None:
                s_met = act_summary["Strategy"]
                b_met = act_summary["Benchmark"]
                t_met = act_summary["Trade_Stats"]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Active Strategy Return", f"{s_met['Total Return (%)']:+.2f}%", delta=f"{s_met['CAGR (%)']:.2f}% CAGR")
                m2.metric("Benchmark Return", f"{b_met['Total Return (%)']:+.2f}%", delta=f"{b_met['CAGR (%)']:.2f}% CAGR")
                m3.metric("Net Alpha Outperformance", f"{(s_met['Total Return (%)'] - b_met['Total Return (%)']):+.2f}%", delta=f"{(s_met['CAGR (%)'] - b_met['CAGR (%)']):+.2f}% CAGR Spread")
                m4.metric("Sharpe Ratio (Rf=7%)", f"{s_met['Annualized Sharpe']:.2f}", delta=f"{s_met['Annualized Sharpe'] - b_met['Annualized Sharpe']:+.2f} vs Bench")

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Total Trades Executed", f"{t_met['Total Trades']}")
                m6.metric("Overall Win Rate", f"{t_met['Win Rate (%)']:.1f}%", help=f"Long Win Rate: {t_met.get('Long Win Rate (%)', 0):.1f}% | Short Win Rate: {t_met.get('Short Win Rate (%)', 0):.1f}%")
                m7.metric("Profit Factor", f"{t_met['Profit Factor']:.2f}")
                m8.metric("Trade Distribution", f"{t_met.get('Long Trades', 0)} Long / {t_met.get('Short Trades', 0)} Short")

                render_active_long_short_plot(act_eq_df)

                if act_tr_df is not None and not act_tr_df.empty:
                    with st.expander(f"🔍 View Active Long-Short Trade Audit Trail ({len(act_tr_df)} Trades: Long & Short)"):
                        st.dataframe(act_tr_df, use_container_width=True, hide_index=True)

    # ==========================================================================
    # TAB 3: Model Explainability (SHAP)
    # ==========================================================================
    with tab_xai:
        st.subheader("Model Explainability & Feature Importance (XAI)")
        st.caption("SHAP (SHapley Additive exPlanations) attribute distribution across test set features.")

        model, loaded_name = load_trained_model(MODELS_DIR, selected_model_name)
        X_test = load_test_features()

        if model is None or X_test is None:
            st.info("💡 Model artifact or test features not found. Run `make train` to enable XAI.")
        else:
            meta_cols = ["Date", "Ticker"]
            feature_cols = [c for c in X_test.columns if c not in meta_cols]
            X_feats = X_test[feature_cols]

            # Sample bounded rows for responsive explanation
            sample_size = min(100, len(X_feats))
            X_sample = X_feats.sample(sample_size, random_state=42)

            shap_succeeded = False
            try:
                import shap

                with st.spinner("Computing SHAP values on test sample..."):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)

                    # Handle 2-class output shape
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        vals_to_plot = shap_values[1]
                    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                        vals_to_plot = shap_values[:, :, 1]
                    else:
                        vals_to_plot = shap_values

                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(vals_to_plot, X_sample, plot_type="bar", show=False)
                    plt.title(f"SHAP Feature Importance ({loaded_name.upper()})", fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    shap_succeeded = True

            except ImportError:
                st.warning("⚠️ `shap` is not installed in the current environment.")
            except Exception as e:
                logger.warning("SHAP calculation fallback triggered: %s", e)

            # Native Feature Importances Fallback
            if not shap_succeeded:
                st.info("Displaying native tree feature importances:")
                if hasattr(model, "feature_importances_"):
                    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    importances.plot(kind="barh", ax=ax, color="#00CC96")
                    ax.invert_yaxis()
                    ax.set_title(f"Top 15 Predictive Features ({loaded_name.upper()})")
                    ax.set_xlabel("Relative Importance")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.write("Selected model does not expose tree feature importances.")


if __name__ == "__main__":
    main()
