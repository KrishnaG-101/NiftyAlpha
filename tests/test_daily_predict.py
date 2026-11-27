"""
Unit tests for scripts/daily_predict.py
======================================
Validates automated daily cron prediction, inference execution, order manifest
formatting, stop-loss / take-profit calculations, and artifact persistence.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from scripts.daily_predict import DailyPredictor


@pytest.fixture
def mock_predict_env(tmp_path: Path):
    """Sets up a minimal test environment with features, prices, and a model."""
    processed_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    processed_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)

    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

    # Feature records
    feat_rows = []
    price_rows = []
    for d in dates:
        for t in tickers:
            feat_rows.append({
                "Date": d,
                "Ticker": t,
                "rsi_14": np.random.uniform(-2, 2),
                "bollinger_pct_b": np.random.uniform(0, 1),
                "ret_20d": np.random.uniform(-0.05, 0.05),
            })
            price_rows.append({
                "Date": d,
                "Ticker": t,
                "Close": np.random.uniform(1000, 3000),
            })

    X_test = pd.DataFrame(feat_rows)
    X_test.to_parquet(processed_dir / "X_test.parquet", index=False)

    raw_nse = pd.DataFrame(price_rows)
    raw_nse.to_parquet(raw_dir / "nse_equities.parquet", index=False)

    # Train and serialize simple model
    feature_cols = ["rsi_14", "bollinger_pct_b", "ret_20d"]
    model = DecisionTreeClassifier(max_depth=2, random_state=42)
    model.fit(X_test[feature_cols], np.random.choice([0, 1], size=len(X_test)))
    joblib.dump(model, models_dir / "xgboost_model.joblib")

    return {
        "processed_dir": processed_dir,
        "raw_dir": raw_dir,
        "models_dir": models_dir,
        "reports_dir": reports_dir,
        "latest_date": dates[-1].strftime("%Y-%m-%d"),
    }


def test_daily_predictor_unit_execution(mock_predict_env):
    """Verify DailyPredictor generates structured order files and correct targets."""
    env = mock_predict_env
    predictor = DailyPredictor(
        processed_dir=env["processed_dir"],
        raw_dir=env["raw_dir"],
        models_dir=env["models_dir"],
        reports_dir=env["reports_dir"],
        model_name="xgboost",
        threshold=0.50,
    )

    result = predictor.run()

    assert result["date"] == env["latest_date"]
    assert result["universe_count"] == 3
    assert "buy_count" in result
    assert isinstance(result["orders"], pd.DataFrame)
    assert len(result["orders"]) == 3

    # Check dated files
    dated_csv = Path(result["dated_csv"])
    dated_json = Path(result["dated_json"])
    assert dated_csv.exists()
    assert dated_json.exists()

    # Check latest files
    latest_csv = Path(result["latest_csv"])
    latest_json = Path(result["latest_json"])
    assert latest_csv.exists()
    assert latest_json.exists()

    # Verify order structure
    df = pd.read_csv(dated_csv)
    expected_cols = [
        "signal_date",
        "execution_date",
        "ticker",
        "action",
        "model_name",
        "confidence_score",
        "confidence_pct",
        "threshold_applied",
        "last_close_inr",
        "execution_directive",
        "target_allocation_pct",
        "stop_loss_inr",
        "take_profit_inr",
        "max_holding_trading_days",
    ]
    for c in expected_cols:
        assert c in df.columns

    # Verify risk controls (SL = 92% of close, TP = 115% of close)
    for _, row in df.iterrows():
        if pd.notna(row["last_close_inr"]):
            px = row["last_close_inr"]
            sl = row["stop_loss_inr"]
            tp = row["take_profit_inr"]
            assert pytest.approx(sl, abs=0.05) == px * 0.92
            assert pytest.approx(tp, abs=0.05) == px * 1.15

    # Verify JSON structure
    with open(dated_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "metadata" in data
    assert "orders" in data
    assert data["metadata"]["universe_count"] == 3
    assert len(data["orders"]) == 3


def test_daily_predictor_target_date_filter(mock_predict_env):
    """Verify target_date argument filters by specified historical date."""
    env = mock_predict_env
    predictor = DailyPredictor(
        processed_dir=env["processed_dir"],
        raw_dir=env["raw_dir"],
        models_dir=env["models_dir"],
        reports_dir=env["reports_dir"],
        model_name="xgboost",
        threshold=0.50,
    )

    # Valid target date
    result = predictor.run(target_date="2024-01-01")
    assert result["date"] == "2024-01-01"

    # Nonexistent date should raise ValueError
    with pytest.raises(ValueError, match="No feature records found"):
        predictor.run(target_date="2010-01-01")


def test_daily_predictor_missing_feature_data(tmp_path: Path):
    """Verify DailyPredictor raises FileNotFoundError when feature files are missing."""
    predictor = DailyPredictor(
        processed_dir=tmp_path / "empty_proc",
        raw_dir=tmp_path / "raw",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )
    with pytest.raises(FileNotFoundError, match="Processed feature matrix not found"):
        predictor.run()
