"""
Unit tests for dashboard/app.py and main.py
===========================================
Validates dashboard loading functions, metric calculations, and CLI routing.
"""

from pathlib import Path
import pandas as pd
import pytest

from dashboard.app import (
    load_equity_curve,
    load_trade_log,
    load_metrics_report,
    load_test_features,
    load_raw_equities,
    load_trained_model,
    load_vector_comparison,
    load_vector_equity_curves,
    load_active_long_short_data,
    load_latest_daily_orders,
)
from main import run_fetch, run_features, run_train, run_simulate


def test_dashboard_loaders_nonexistent(tmp_path: Path):
    """Verify dashboard loaders handle missing files gracefully by returning None."""
    assert load_equity_curve(reports_dir=tmp_path) is None
    assert load_trade_log(reports_dir=tmp_path) is None
    assert load_metrics_report(reports_dir=tmp_path) is None
    assert load_test_features(processed_dir=tmp_path) is None
    assert load_raw_equities(raw_dir=tmp_path) is None
    assert load_vector_comparison(reports_dir=tmp_path) is None
    assert load_vector_equity_curves(reports_dir=tmp_path) is None
    act_eq, act_tr, act_sm = load_active_long_short_data(reports_dir=tmp_path)
    assert act_eq is None and act_tr is None and act_sm is None
    daily_orders, daily_meta = load_latest_daily_orders(reports_dir=tmp_path)
    assert daily_orders is None and daily_meta is None
    model, name = load_trained_model(models_dir=tmp_path)
    assert model is None
    assert name == "none"


def test_dashboard_loaders_with_data(tmp_path: Path):
    """Verify dashboard loaders load existing files properly."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)

    # Mock equity curve
    equity_df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02"],
        "Strategy_Value": [1000000.0, 1010000.0],
        "Benchmark_Value": [1000000.0, 1005000.0],
    })
    equity_df.to_csv(reports_dir / "equity_curve.csv", index=False)

    loaded_eq = load_equity_curve(reports_dir=reports_dir)
    assert loaded_eq is not None
    assert len(loaded_eq) == 2
    assert pd.api.types.is_datetime64_any_dtype(loaded_eq["Date"])

    # Mock active long short data
    equity_df.to_csv(reports_dir / "active_long_short_equity.csv", index=False)
    pd.DataFrame({"Ticker": ["TCS.NS"], "Type": ["LONG"], "Return_Pct": [5.2]}).to_csv(reports_dir / "active_long_short_trades.csv", index=False)
    import json
    with open(reports_dir / "active_long_short_summary.json", "w") as f:
        json.dump({"Strategy": {"Total Return (%)": 36.82}}, f)

    act_eq, act_tr, act_sm = load_active_long_short_data(reports_dir=reports_dir)
    assert act_eq is not None
    assert act_tr is not None
    assert act_sm is not None
    assert len(act_eq) == 2

    # Mock daily orders
    orders_df = pd.DataFrame({
        "ticker": ["INFY.NS"],
        "action": ["BUY"],
        "confidence_score": [0.65],
    })
    orders_df.to_csv(reports_dir / "latest_daily_orders.csv", index=False)
    with open(reports_dir / "latest_daily_orders.json", "w") as f:
        json.dump({"metadata": {"signal_date": "2024-01-02", "buy_orders_count": 1}, "orders": []}, f)

    daily_df, daily_meta = load_latest_daily_orders(reports_dir=reports_dir)
    assert daily_df is not None
    assert len(daily_df) == 1
    assert daily_meta["buy_orders_count"] == 1
