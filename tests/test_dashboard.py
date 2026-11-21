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
)
from main import run_fetch, run_features, run_train, run_simulate


def test_dashboard_loaders_nonexistent(tmp_path: Path):
    """Verify dashboard loaders handle missing files gracefully by returning None."""
    assert load_equity_curve(reports_dir=tmp_path) is None
    assert load_trade_log(reports_dir=tmp_path) is None
    assert load_metrics_report(reports_dir=tmp_path) is None
    assert load_test_features(processed_dir=tmp_path) is None
    assert load_raw_equities(raw_dir=tmp_path) is None
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
