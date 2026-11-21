"""
Unit tests for src/fetch_data.py
================================
Validates data universe configurations, DataFrame cleaners, and caching mechanics.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.fetch_data import (
    NSE_EQUITY_TICKERS,
    BSE_EQUITY_TICKERS,
    INDEX_TICKERS,
    COMMODITY_FX_TICKERS,
    FRED_SERIES_IDS,
    TARGET_COLUMNS,
    _clean_yfinance_dataframe,
    DataIngestionPipeline,
)


def test_universe_sizes_and_mappings():
    """Verify universe definitions match specification."""
    assert len(NSE_EQUITY_TICKERS) == 20
    assert len(BSE_EQUITY_TICKERS) == 20
    assert len(INDEX_TICKERS) == 3
    assert len(COMMODITY_FX_TICKERS) == 3
    assert len(FRED_SERIES_IDS) == 2

    # Verify 1-to-1 mapping from .NS to .BO
    for nse, bse in zip(NSE_EQUITY_TICKERS, BSE_EQUITY_TICKERS):
        assert nse.replace(".NS", ".BO") == bse


def test_clean_yfinance_dataframe_multiindex():
    """Test standardizing DataFrame when yfinance returns MultiIndex columns."""
    dates = pd.date_range("2024-01-01", periods=3, tz="Asia/Kolkata")
    tuples = [
        ("Open", "RELIANCE.NS"),
        ("High", "RELIANCE.NS"),
        ("Low", "RELIANCE.NS"),
        ("Close", "RELIANCE.NS"),
        ("Adj Close", "RELIANCE.NS"),
        ("Volume", "RELIANCE.NS"),
    ]
    cols = pd.MultiIndex.from_tuples(tuples, names=["Price", "Ticker"])
    data = np.random.rand(3, 6)
    raw_df = pd.DataFrame(data, index=dates, columns=cols)

    cleaned = _clean_yfinance_dataframe(raw_df, "RELIANCE.NS")

    assert list(cleaned.columns) == TARGET_COLUMNS
    assert len(cleaned) == 3
    assert (cleaned["Ticker"] == "RELIANCE.NS").all()
    # Ensure timezone was stripped
    assert cleaned["Date"].dt.tz is None


def test_clean_yfinance_dataframe_empty():
    """Test cleaner on empty input."""
    empty_df = pd.DataFrame()
    cleaned = _clean_yfinance_dataframe(empty_df, "TEST.NS")
    assert list(cleaned.columns) == TARGET_COLUMNS
    assert len(cleaned) == 0


def test_caching_skips_existing_files(tmp_path: Path):
    """Test that existing parquet files are not re-downloaded unless force=True."""
    pipeline = DataIngestionPipeline(raw_data_dir=tmp_path)
    nse_target = tmp_path / "nse_equities.parquet"

    # Create dummy parquet
    dummy_df = pd.DataFrame({
        "Date": [pd.Timestamp("2024-01-01")],
        "Ticker": ["RELIANCE.NS"],
        "Open": [2500.0],
        "High": [2550.0],
        "Low": [2490.0],
        "Close": [2540.0],
        "Adj Close": [2540.0],
        "Volume": [1000000.0],
    })
    dummy_df.to_parquet(nse_target, index=False)

    # Calling fetch_nse_equities without force should return immediately
    result_path = pipeline.fetch_nse_equities(force=False)
    assert result_path == nse_target

    loaded = pd.read_parquet(result_path)
    assert len(loaded) == 1
    assert loaded["Ticker"].iloc[0] == "RELIANCE.NS"
