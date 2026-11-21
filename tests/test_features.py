"""
Unit tests for src/features.py
==============================
Validates technical indicator calculations, target variables, categorical encodings,
leakage-free temporal splitting, and standard scaling.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features import (
    _calculate_rsi,
    _calculate_macd,
    _calculate_atr,
    _calculate_bollinger_pct_b,
    _calculate_obv,
    compute_ticker_features,
    add_categorical_and_dummy_features,
    FeatureEngineeringPipeline,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
)


def _generate_synthetic_stock_data(n_days: int = 150, ticker: str = "RELIANCE.NS", start_date: str = "2020-01-01") -> pd.DataFrame:
    """Helper to generate consistent OHLCV synthetic data with context columns."""
    dates = pd.date_range(start_date, periods=n_days, freq="B")
    np.random.seed(42)
    # Generate geometric random walk
    returns = np.random.normal(0.001, 0.02, size=n_days)
    close = 2000.0 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, size=n_days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, size=n_days)))
    open_p = (high + low) / 2.0
    volume = np.random.uniform(500000, 2000000, size=n_days)

    df = pd.DataFrame({
        "Date": dates,
        "Ticker": ticker,
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": close,
        "Adj Close": close,
        "Volume": volume,
        "Volume_BSE": volume * np.random.uniform(0.1, 0.3, size=n_days),
        "Nifty_Close": 15000.0 * np.cumprod(1 + np.random.normal(0.0005, 0.01, size=n_days)),
        "Sensex_Close": 50000.0 * np.cumprod(1 + np.random.normal(0.0005, 0.01, size=n_days)),
        "India_VIX": np.random.uniform(12.0, 25.0, size=n_days),
        "Brent_Close": 75.0 * np.cumprod(1 + np.random.normal(0.0005, 0.02, size=n_days)),
        "USDINR_Close": 82.0 * np.cumprod(1 + np.random.normal(0.0001, 0.005, size=n_days)),
    })
    return df


def test_technical_indicators():
    """Verify numeric calculations for RSI, MACD, ATR, Bollinger %B, and OBV."""
    df = _generate_synthetic_stock_data(n_days=100)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    rsi = _calculate_rsi(close, 14)
    assert len(rsi) == 100
    # RSI bounded between 0 and 100
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0.0).all() and (valid_rsi <= 100.0).all()

    macd, signal, hist = _calculate_macd(close, 12, 26, 9)
    assert len(macd) == 100
    assert len(signal) == 100
    assert len(hist) == 100
    # hist = macd - signal
    np.testing.assert_allclose(hist.values, (macd - signal).values, rtol=1e-5)

    atr = _calculate_atr(high, low, close, 14)
    assert (atr.dropna() > 0.0).all()

    pct_b = _calculate_bollinger_pct_b(close, 20)
    assert len(pct_b) == 100

    obv = _calculate_obv(close, volume)
    assert len(obv) == 100


def test_target_variables():
    """Verify 20-trading-day forward return shift(-20) and binary growth label."""
    df = _generate_synthetic_stock_data(n_days=50)
    featured = compute_ticker_features(df)

    assert "target_30d_return" in featured.columns
    assert "is_positive_growth" in featured.columns

    # Rows from index 0 to 29 must have targets
    assert not featured["target_30d_return"].iloc[0:30].isna().any()
    # The last 20 rows must be NaN due to shift(-20)
    assert featured["target_30d_return"].iloc[-20:].isna().all()

    # Verify return formula: (Close[t+20] - Close[t]) / Close[t]
    expected_ret_0 = (featured["Close"].iloc[20] - featured["Close"].iloc[0]) / featured["Close"].iloc[0]
    np.testing.assert_almost_equal(featured["target_30d_return"].iloc[0], expected_ret_0)

    # Verify binary label
    expected_binary = int(expected_ret_0 > 0.0)
    assert featured["is_positive_growth"].iloc[0] == expected_binary


def test_categorical_and_dummy_encodings():
    """Verify calendar encodings, budget indicator, VIX regime, and dummy dimensions."""
    df = _generate_synthetic_stock_data(n_days=30, start_date="2022-02-01")
    # Force some VIX values > 20
    df["India_VIX"] = [25.0 if i % 2 == 0 else 15.0 for i in range(len(df))]

    encoded = add_categorical_and_dummy_features(df)

    assert "is_budget_month" in encoded.columns
    # In February (first 20 business days of Feb 2022), is_budget_month must be 1; in March it must be 0
    assert (encoded["is_budget_month"].iloc[0:20] == 1).all()
    assert (encoded["is_budget_month"].iloc[20:30] == 0).all()

    assert "vix_regime_high" in encoded.columns
    assert (encoded["vix_regime_high"].iloc[0] == 1)
    assert (encoded["vix_regime_high"].iloc[1] == 0)

    # Check 12 month dummy columns and 5 day dummy columns exist
    for m in range(1, 13):
        assert f"month_{m}" in encoded.columns
    for d in range(5):
        assert f"day_{d}" in encoded.columns


def test_temporal_split_and_scaling(tmp_path: Path):
    """
    Verify strict temporal partitioning and that StandardScaler is fitted
    exclusively on the Train split without future leakage.
    """
    # Create multi-year synthetic dataset from 2020 to 2024
    dates = pd.date_range("2020-01-01", "2024-06-30", freq="B")
    n_days = len(dates)

    t1 = _generate_synthetic_stock_data(n_days=n_days, ticker="STOCK1.NS", start_date="2020-01-01")
    t2 = _generate_synthetic_stock_data(n_days=n_days, ticker="STOCK2.NS", start_date="2020-01-01")
    combined = pd.concat([t1, t2], ignore_index=True)

    pipeline = FeatureEngineeringPipeline(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
    )

    featured = pipeline.process_features(combined)
    results = pipeline.temporal_split_and_scale(featured)

    # Check all 6 output parquet files exist
    for key, path in results.items():
        assert path.exists()

    X_train = pd.read_parquet(results["X_train"])
    y_train = pd.read_parquet(results["y_train"])
    X_val = pd.read_parquet(results["X_val"])
    y_val = pd.read_parquet(results["y_val"])
    X_test = pd.read_parquet(results["X_test"])
    y_test = pd.read_parquet(results["y_test"])

    # Verify temporal bounds
    assert (X_train["Date"] <= pd.Timestamp("2021-12-31")).all()
    assert ((X_val["Date"] >= pd.Timestamp("2022-01-01")) & (X_val["Date"] <= pd.Timestamp("2023-12-31"))).all()
    assert (X_test["Date"] >= pd.Timestamp("2024-01-01")).all()

    # Verify Metadata preservation
    assert "Date" in X_train.columns and "Ticker" in X_train.columns
    assert "Date" in y_train.columns and "Ticker" in y_train.columns

    # Verify Scaling: Train numeric features should have mean ~ 0 and std ~ 1
    for num_col in NUMERIC_FEATURES:
        mean_val = X_train[num_col].mean()
        std_val = X_train[num_col].std()
        np.testing.assert_almost_equal(mean_val, 0.0, decimal=1)
        np.testing.assert_almost_equal(std_val, 1.0, decimal=1)

    # Verify scaler artifact persistence
    assert (tmp_path / "models" / "scaler.joblib").exists()
