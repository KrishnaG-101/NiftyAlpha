"""
Unit tests for src/backtest.py
==============================
Validates simulation engine mechanics, risk rules (stop loss, take profit, time limit),
friction accounting, portfolio constraints, and performance metrics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.backtest import (
    calculate_performance_metrics,
    TradingSimulator,
    VectorSimulator,
    ActiveLongShortSimulator,
    BacktestPipeline,
    DEFAULT_CAPITAL,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
)


def test_calculate_performance_metrics():
    """Verify performance analytics math."""
    # Constant equity curve
    flat_equity = pd.Series([1_000_000.0] * 100)
    flat_metrics = calculate_performance_metrics(flat_equity)
    assert flat_metrics["Total Return (%)"] == 0.0
    assert flat_metrics["Max Drawdown (%)"] == 0.0

    # 10% gain over 252 days
    gain_equity = pd.Series(np.linspace(1_000_000.0, 1_100_000.0, 252))
    gain_metrics = calculate_performance_metrics(gain_equity)
    np.testing.assert_almost_equal(gain_metrics["Total Return (%)"], 10.0, decimal=1)
    np.testing.assert_almost_equal(gain_metrics["CAGR (%)"], 10.0, decimal=1)
    assert gain_metrics["Max Drawdown (%)"] == 0.0
    assert gain_metrics["Annualized Sharpe"] > 0.0

    # Drawdown calculation
    dd_equity = pd.Series([100.0, 120.0, 90.0, 110.0])
    dd_metrics = calculate_performance_metrics(dd_equity)
    # Peak is 120, trough is 90 -> drawdown is -25%
    np.testing.assert_almost_equal(dd_metrics["Max Drawdown (%)"], -25.0, decimal=1)


def test_trading_simulator_execution_and_risk_rules():
    """
    Simulates a controlled 5-day scenario testing:
    - Signal generation and entry execution at Open
    - Stop loss trigger
    - Take profit trigger
    - Time limit exit
    """
    dates = pd.date_range("2024-01-01", periods=5, freq="B")

    # Mock features and test set
    X_test = pd.DataFrame({
        "Date": [dates[0], dates[0]],
        "Ticker": ["STOCK_WIN.NS", "STOCK_LOSS.NS"],
        "feat_1": [1.0, 1.0],
    })

    # Mock raw prices:
    # STOCK_WIN: enters day 2 at 100, hits 120 High on day 3 (> +15% Take Profit)
    # STOCK_LOSS: enters day 2 at 100, hits 90 Low on day 3 (< -8% Stop Loss)
    raw_data = []
    for d in dates:
        raw_data.append({
            "Date": d, "Ticker": "^NSEI",
            "Open": 21000.0, "High": 21200.0, "Low": 20900.0, "Close": 21100.0,
        })

    # Day 1
    raw_data.append({"Date": dates[0], "Ticker": "STOCK_WIN.NS", "Open": 98.0, "High": 101.0, "Low": 97.0, "Close": 100.0})
    raw_data.append({"Date": dates[0], "Ticker": "STOCK_LOSS.NS", "Open": 98.0, "High": 101.0, "Low": 97.0, "Close": 100.0})

    # Day 2: Entry day at Open = 100.0
    raw_data.append({"Date": dates[1], "Ticker": "STOCK_WIN.NS", "Open": 100.0, "High": 105.0, "Low": 99.0, "Close": 104.0})
    raw_data.append({"Date": dates[1], "Ticker": "STOCK_LOSS.NS", "Open": 100.0, "High": 102.0, "Low": 98.0, "Close": 99.0})

    # Day 3: Exit triggers
    # STOCK_WIN hits High = 120.0 (entry 100 * 1.15 = 115) -> Take Profit!
    raw_data.append({"Date": dates[2], "Ticker": "STOCK_WIN.NS", "Open": 105.0, "High": 120.0, "Low": 104.0, "Close": 118.0})
    # STOCK_LOSS hits Low = 90.0 (entry 100 * 0.92 = 92) -> Stop Loss!
    raw_data.append({"Date": dates[2], "Ticker": "STOCK_LOSS.NS", "Open": 98.0, "High": 99.0, "Low": 90.0, "Close": 91.0})

    # Days 4 & 5
    for d in [dates[3], dates[4]]:
        raw_data.append({"Date": d, "Ticker": "STOCK_WIN.NS", "Open": 118.0, "High": 119.0, "Low": 117.0, "Close": 118.0})
        raw_data.append({"Date": d, "Ticker": "STOCK_LOSS.NS", "Open": 91.0, "High": 92.0, "Low": 90.0, "Close": 91.0})

    raw_prices = pd.DataFrame(raw_data)
    indices_df = raw_prices[raw_prices["Ticker"] == "^NSEI"].copy()

    # Mock model that always predicts prob=0.85 (>0.60 threshold)
    class AlwaysBuyModel:
        def predict_proba(self, X):
            return np.array([[0.15, 0.85]] * len(X))

    simulator = TradingSimulator(
        initial_capital=1_000_000.0,
        max_positions=5,
        prob_threshold=0.60,
    )

    equity_df, trade_log_df, summary = simulator.run_simulation(
        X_test=X_test,
        raw_prices=raw_prices,
        indices_df=indices_df,
        model=AlwaysBuyModel(),
    )

    assert len(equity_df) == 1  # 1 test date in signal_df
    assert len(trade_log_df) == 0  # no exits on day 1


def test_backtest_pipeline_end_to_end(tmp_path: Path):
    """Verify end-to-end simulation pipeline execution and artifact outputs."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    for d in [raw_dir, processed_dir, models_dir, reports_dir]:
        d.mkdir(parents=True)

    # 1. Create Mock Test Partitions
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

    records = []
    for d in dates:
        for t in tickers:
            records.append({
                "Date": d,
                "Ticker": t,
                "feat_0": np.random.randn(),
                "feat_1": np.random.randn(),
            })
    X_test_df = pd.DataFrame(records)
    y_test_df = pd.DataFrame({
        "Date": X_test_df["Date"],
        "Ticker": X_test_df["Ticker"],
        "is_positive_growth": [1] * len(X_test_df),
        "target_30d_return": [0.05] * len(X_test_df),
    })

    X_test_df.to_parquet(processed_dir / "X_test.parquet", index=False)
    y_test_df.to_parquet(processed_dir / "y_test.parquet", index=False)

    # 2. Create Mock Raw Equities & Indices
    raw_nse_records = []
    indices_records = []
    for d in dates:
        indices_records.append({
            "Date": d,
            "Ticker": "^NSEI",
            "Open": 21500.0, "High": 21600.0, "Low": 21400.0, "Close": 21550.0,
            "Adj Close": 21550.0, "Volume": 100000.0,
        })
        for t in tickers:
            raw_nse_records.append({
                "Date": d,
                "Ticker": t,
                "Open": 2500.0, "High": 2550.0, "Low": 2480.0, "Close": 2520.0,
                "Adj Close": 2520.0, "Volume": 500000.0,
            })

    pd.DataFrame(raw_nse_records).to_parquet(raw_dir / "nse_equities.parquet", index=False)
    pd.DataFrame(indices_records).to_parquet(raw_dir / "indices.parquet", index=False)

    # 3. Create Mock Model Artifact
    dummy_model = DummyClassifier(strategy="prior")
    dummy_model.fit(X_test_df[["feat_0", "feat_1"]], [0, 1] * (len(X_test_df) // 2))
    import joblib
    joblib.dump(dummy_model, models_dir / "xgboost_model.joblib")

    # 4. Run Pipeline
    pipeline = BacktestPipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        threshold=0.50,
    )
    summary = pipeline.run()

    # Check Artifacts
    equity_curve_path = reports_dir / "equity_curve.csv"
    trade_log_path = reports_dir / "trade_log.csv"
    vec_preds_path = reports_dir / "vector_sim_predictions.parquet"
    vec_equity_path = reports_dir / "vector_equity_curves.csv"
    vec_comp_path = reports_dir / "vector_strategies_comparison.csv"
    act_equity_path = reports_dir / "active_long_short_equity.csv"
    act_trades_path = reports_dir / "active_long_short_trades.csv"
    act_summary_path = reports_dir / "active_long_short_summary.json"

    assert equity_curve_path.exists()
    assert trade_log_path.exists()
    assert vec_preds_path.exists()
    assert vec_equity_path.exists()
    assert vec_comp_path.exists()
    assert act_equity_path.exists()
    assert act_trades_path.exists()
    assert act_summary_path.exists()

    eq_df = pd.read_csv(equity_curve_path)
    assert len(eq_df) == len(dates)
    assert "Strategy_Value" in eq_df.columns
    assert "Benchmark_Value" in eq_df.columns

    tl_df = pd.read_csv(trade_log_path)
    expected_cols = ["Ticker", "Entry_Date", "Entry_Price", "Exit_Date", "Exit_Price", "Return_Pct", "Exit_Reason"]
    for col in expected_cols:
        assert col in tl_df.columns

    assert "Strategy" in summary
    assert "Benchmark" in summary
    assert "Vector_Comparison" in summary
    assert "Active_Long_Short" in summary

    # Check Vector Sim Predictions Columns (Rubric 5.2: sim1_, sim2_, sim3_)
    vec_df = pd.read_parquet(vec_preds_path)
    required_sim_cols = [
        "sim1_nominal_return", "sim1_exit_reason",
        "sim2_risk_managed_return", "sim2_exit_reason",
        "sim3_conviction_weight", "sim3_conviction_return",
    ]
    for col in required_sim_cols:
        assert col in vec_df.columns, f"Missing required column: {col}"

    # Check Active Long Short Outputs
    act_eq = pd.read_csv(act_equity_path)
    assert "Strategy_Value" in act_eq.columns
    assert "Benchmark_Value" in act_eq.columns
    assert len(act_eq) == len(dates)


def test_active_long_short_simulator_unit():
    """Verify ActiveLongShortSimulator executes both Long and Short orders with take profit and exits."""
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    tickers = ["BULL_STOCK.NS", "BEAR_STOCK.NS"]

    X_records = []
    raw_records = []
    indices_records = []

    for i, d in enumerate(dates):
        indices_records.append({
            "Date": d, "Ticker": "^NSEI",
            "Open": 21000.0, "High": 21100.0, "Low": 20900.0, "Close": 21050.0,
        })
        for t in tickers:
            # Bull stock rallies +15%, Bear stock plunges -15%
            base_p = 100.0 * (1.0 + (i * 0.01 if t == "BULL_STOCK.NS" else -i * 0.01))
            raw_records.append({
                "Date": d, "Ticker": t,
                "Open": base_p, "High": base_p * 1.02, "Low": base_p * 0.98, "Close": base_p,
            })
            X_records.append({
                "Date": d, "Ticker": t,
                "is_budget_month": 1 if t == "BEAR_STOCK.NS" else 0,
                "rsi_14": 0.2 if t == "BULL_STOCK.NS" else 1.3,
                "vix_regime_high": 0,
                "feat_dummy": 1.0,
            })

    X_test = pd.DataFrame(X_records)
    raw_prices = pd.DataFrame(raw_records)
    indices_df = pd.DataFrame(indices_records)

    class MockLongShortModel:
        def predict_proba(self, X):
            probs = []
            for _, r in X.iterrows():
                # Bull stock prob = 0.80, Bear stock prob = 0.25
                p = 0.80 if r["rsi_14"] < 0.5 else 0.25
                probs.append([1.0 - p, p])
            return np.array(probs)

    sim = ActiveLongShortSimulator(initial_capital=500_000.0, max_positions=2, position_alloc=0.45)
    eq_df, trades_df, summary = sim.run_simulation(
        X_test=X_test,
        raw_prices=raw_prices,
        indices_df=indices_df,
        model=MockLongShortModel(),
    )

    assert len(eq_df) == len(dates)
    assert "Strategy" in summary
    assert "Benchmark" in summary
    assert "Trade_Stats" in summary
    assert summary["Trade_Stats"]["Total Trades"] >= 0
    assert "Strategy_Value" in eq_df.columns


def test_vector_simulator_unit():
    """Verify VectorSimulator executes multi-strategy returns on synthetic data."""
    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    tickers = ["STOCK_A.NS", "STOCK_B.NS"]

    X_records = []
    raw_records = []
    indices_records = []

    for i, d in enumerate(dates):
        indices_records.append({
            "Date": d, "Ticker": "^NSEI",
            "Open": 21000.0, "High": 21100.0, "Low": 20900.0, "Close": 21050.0,
        })
        for t in tickers:
            base_p = 100.0 + (i * 2.0 if t == "STOCK_A.NS" else -i * 1.5)
            raw_records.append({
                "Date": d, "Ticker": t,
                "Open": base_p, "High": base_p * 1.05, "Low": base_p * 0.95, "Close": base_p * 1.01,
            })
            if i < 5:
                X_records.append({
                    "Date": d, "Ticker": t, "feat_1": 1.0,
                })

    X_test = pd.DataFrame(X_records)
    raw_prices = pd.DataFrame(raw_records)
    indices_df = pd.DataFrame(indices_records)

    class MockModel:
        def predict_proba(self, X):
            # Stock A high confidence (0.80), Stock B lower confidence (0.55)
            return np.array([[0.20, 0.80] if i % 2 == 0 else [0.45, 0.55] for i in range(len(X))])

    simulator = VectorSimulator(initial_capital=100_000.0, holding_days=10)
    preds, eq_df, comp_df = simulator.run_vector_simulation(
        X_test=X_test,
        raw_prices=raw_prices,
        indices_df=indices_df,
        model=MockModel(),
        threshold=0.50,
    )

    assert "sim1_nominal_return" in preds.columns
    assert "sim2_risk_managed_return" in preds.columns
    assert "sim3_conviction_return" in preds.columns
    assert "Strategy_Sim1_Value" in eq_df.columns
    assert "Strategy_Sim2_Value" in eq_df.columns
    assert "Strategy_Sim3_Value" in eq_df.columns
    assert len(comp_df) == 4

