"""
NiftyAlpha - Multi-Source Data Ingestion Module (Phase 2)
=========================================================
Production-grade data ingestion pipeline for the Indian Equity Market and
macroeconomic drivers.

Data Universes:
1. NSE Equities: Full Nifty 50 constituents (.NS)
2. BSE Equities: Corresponding BSE listings (.BO)
3. Indices: Nifty 50 (^NSEI), BSE Sensex (^BSESN), India VIX (^INDIAVIX)
4. Macro/Commodities: Brent Crude (BZ=F), Gold (GC=F), USD/INR (INR=X)
5. FRED Macro Series: US 10Y Yield (DGS10), US Fed Funds Rate (FEDFUNDS)

Features:
- Robust retry logic with exponential backoff
- Rate limiting to respect API policies
- Parquet storage with fast columnar caching
- CLI interface supporting cache overrides (--force)
"""

import argparse
from datetime import datetime
import io
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NiftyAlpha.DataIngestion")

# Resolve Project Root and Default Storage Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ==============================================================================
# Universe Definitions
# ==============================================================================

NSE_EQUITY_TICKERS: List[str] = [
    # Top Mega-Caps (Original 20)
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "TATASTEEL.NS", "M&M.NS", "HCLTECH.NS",
    # Additional Nifty 50 Constituents (Expanded 30)
    "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "BPCL.NS", "COALINDIA.NS",
    "IOC.NS", "ULTRACEMCO.NS", "GRASIM.NS", "HINDALCO.NS", "JSWSTEEL.NS",
    "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "BRITANNIA.NS",
    "NESTLEIND.NS", "TATACONSUM.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
    "BAJAJFINSV.NS", "SHRIRAMFIN.NS", "INDUSINDBK.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "TECHM.NS", "WIPRO.NS", "ADANIENT.NS", "ADANIPORTS.NS", "TRENT.NS",
]

BSE_EQUITY_TICKERS: List[str] = [
    ticker.replace(".NS", ".BO") for ticker in NSE_EQUITY_TICKERS
]

INDEX_TICKERS: List[str] = [
    "^NSEI",       # Nifty 50
    "^BSESN",      # BSE Sensex
    "^INDIAVIX",   # India VIX
]

COMMODITY_FX_TICKERS: List[str] = [
    "BZ=F",        # Brent Crude Oil Futures
    "GC=F",        # Gold Futures
    "INR=X",       # USD / INR Spot Exchange Rate
]

FRED_SERIES_IDS: List[str] = [
    "DGS10",       # Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity
    "FEDFUNDS",    # Federal Funds Effective Rate
]

# Standard column schema for quantitative consistency
TARGET_COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


# ==============================================================================
# Helper Functions & Cleaners
# ==============================================================================

def _clean_yfinance_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Cleans and standardizes raw DataFrame from yfinance into a canonical schema.
    Strips timezones, flattens MultiIndex columns, and normalizes date formats.
    """
    if df.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    cleaned = df.copy()

    # Flatten MultiIndex columns if returned by newer yfinance versions
    if isinstance(cleaned.columns, pd.MultiIndex):
        cleaned.columns = cleaned.columns.get_level_values(0)

    # Normalize DatetimeIndex (strip timezone to avoid tz-aware/naive merge conflicts)
    if isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = cleaned.index.tz_localize(None)

    cleaned = cleaned.reset_index()

    # Standardize Date column naming
    for col in cleaned.columns:
        if str(col).lower() in ["date", "datetime", "index"]:
            cleaned = cleaned.rename(columns={col: "Date"})
            break

    # Ensure Date column is standard datetime without hours
    cleaned["Date"] = pd.to_datetime(cleaned["Date"]).dt.normalize()

    # Ensure all expected columns exist
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col not in cleaned.columns:
            if col == "Adj Close" and "Close" in cleaned.columns:
                cleaned["Adj Close"] = cleaned["Close"]
            else:
                cleaned[col] = np.nan

    cleaned["Ticker"] = ticker

    # Select, order, and sort canonical schema
    cleaned = cleaned[TARGET_COLUMNS].sort_values(by=["Date"]).reset_index(drop=True)
    return cleaned


def fetch_ticker_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    request_delay: float = 0.5,
) -> Optional[pd.DataFrame]:
    """
    Fetches historical OHLCV data for a single ticker via yfinance with retry
    and exponential backoff logic.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance is not installed. Please run 'pip install yfinance'.")
        raise

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("Downloading %s (Attempt %d/%d)...", ticker, attempt, max_retries)
            raw_data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
            )

            if raw_data is not None and not raw_data.empty:
                cleaned_data = _clean_yfinance_dataframe(raw_data, ticker)
                logger.info(
                    "Successfully fetched %-14s | %5d rows [%s to %s]",
                    ticker,
                    len(cleaned_data),
                    cleaned_data["Date"].iloc[0].strftime("%Y-%m-%d"),
                    cleaned_data["Date"].iloc[-1].strftime("%Y-%m-%d"),
                )
                time.sleep(request_delay)
                return cleaned_data
            else:
                logger.warning(
                    "Empty dataset received for %s on attempt %d/%d.",
                    ticker,
                    attempt,
                    max_retries,
                )

        except Exception as e:
            logger.warning(
                "Error fetching %s (attempt %d/%d): %s",
                ticker,
                attempt,
                max_retries,
                e,
            )

        if attempt < max_retries:
            sleep_time = backoff_factor ** attempt
            logger.debug("Backing off for %.2f seconds before retrying %s...", sleep_time, ticker)
            time.sleep(sleep_time)

    logger.error("Failed to fetch %s after %d attempts. Skipping.", ticker, max_retries)
    return None


def fetch_fred_series(
    series_id: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    request_delay: float = 0.5,
) -> Optional[pd.DataFrame]:
    """
    Fetches macroeconomic series from FRED via pandas_datareader with a direct
    St. Louis Fed CSV fallback mechanism.
    """
    for attempt in range(1, max_retries + 1):
        df: Optional[pd.DataFrame] = None
        # Attempt 1: Try pandas_datareader if installed
        try:
            import pandas_datareader.data as web
            raw_df = web.DataReader(series_id, "fred", start=start_date, end=end_date)
            if raw_df is not None and not raw_df.empty:
                df = raw_df.reset_index()
                df.rename(columns={"DATE": "Date", "Date": "Date", series_id: "Close"}, inplace=True)
        except Exception as pdr_err:
            logger.debug(
                "pandas_datareader attempt for %s failed (%s). Trying direct FRED endpoint.",
                series_id,
                pdr_err,
            )

        # Attempt 2: Direct St. Louis Fed endpoint fallback
        if df is None or df.empty:
            try:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    raw_df = pd.read_csv(io.StringIO(resp.text))
                    raw_df.rename(columns={"DATE": "Date", series_id: "Close"}, inplace=True)
                    raw_df["Date"] = pd.to_datetime(raw_df["Date"], errors="coerce")
                    # Filter date bounds
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    raw_df = raw_df[(raw_df["Date"] >= start_dt) & (raw_df["Date"] <= end_dt)]
                    # Coerce numeric values (FRED uses '.' for missing values)
                    raw_df["Close"] = pd.to_numeric(raw_df["Close"], errors="coerce")
                    raw_df.dropna(subset=["Close"], inplace=True)
                    df = raw_df
            except Exception as http_err:
                logger.warning(
                    "Direct FRED download failed for %s on attempt %d/%d: %s",
                    series_id,
                    attempt,
                    max_retries,
                    http_err,
                )

        if df is not None and not df.empty:
            df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
            df["Ticker"] = series_id
            df["Open"] = df["Close"]
            df["High"] = df["Close"]
            df["Low"] = df["Close"]
            df["Adj Close"] = df["Close"]
            df["Volume"] = 0.0

            cleaned_df = df[TARGET_COLUMNS].sort_values(by=["Date"]).reset_index(drop=True)
            logger.info(
                "Successfully fetched %-14s | %5d rows [%s to %s]",
                series_id,
                len(cleaned_df),
                cleaned_df["Date"].iloc[0].strftime("%Y-%m-%d"),
                cleaned_df["Date"].iloc[-1].strftime("%Y-%m-%d"),
            )
            time.sleep(request_delay)
            return cleaned_df

        if attempt < max_retries:
            sleep_time = backoff_factor ** attempt
            time.sleep(sleep_time)

    logger.error("Failed to fetch FRED series %s after %d attempts. Skipping.", series_id, max_retries)
    return None


# ==============================================================================
# Pipeline Orchestrator
# ==============================================================================

class DataIngestionPipeline:
    """
    Orchestrates downloading, standardization, caching, and storage of market
    and macroeconomic data universes.
    """

    def __init__(
        self,
        raw_data_dir: Path = DEFAULT_RAW_DATA_DIR,
        start_date: str = "2014-01-01",
        end_date: Optional[str] = None,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime("%Y-%m-%d")

    def _fetch_ticker_batch(self, tickers: List[str]) -> pd.DataFrame:
        """
        Iterates sequentially over a list of tickers, fetching data with rate-limiting
        and aggregate logging.
        """
        collected: List[pd.DataFrame] = []
        total = len(tickers)
        for idx, ticker in enumerate(tickers, start=1):
            logger.info("[%2d/%2d] Ingesting ticker: %s", idx, total, ticker)
            df = fetch_ticker_yfinance(ticker, self.start_date, self.end_date)
            if df is not None and not df.empty:
                collected.append(df)

        if not collected:
            logger.warning("No data retrieved for batch: %s", tickers)
            return pd.DataFrame(columns=TARGET_COLUMNS)

        combined = pd.concat(collected, ignore_index=True)
        return combined.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

    def fetch_nse_equities(self, force: bool = False) -> Path:
        """
        Fetches all 50 Nifty equities, saving to data/raw/nse_equities.parquet.
        """
        target_path = self.raw_data_dir / "nse_equities.parquet"
        if target_path.exists() and not force:
            logger.info("Cached NSE Equities data found at %s. Skipping download.", target_path)
            return target_path

        logger.info("Starting download of NSE Equities (%d tickers)...", len(NSE_EQUITY_TICKERS))
        df = self._fetch_ticker_batch(NSE_EQUITY_TICKERS)
        if not df.empty:
            df.to_parquet(target_path, index=False, engine="pyarrow")
            logger.info("Saved NSE Equities to %s (%d records).", target_path, len(df))
        else:
            logger.error("Failed to produce dataset for NSE Equities.")
        return target_path

    def fetch_bse_equities(self, force: bool = False) -> Path:
        """
        Fetches the corresponding BSE equities (.BO), saving to data/raw/bse_equities.parquet.
        """
        target_path = self.raw_data_dir / "bse_equities.parquet"
        if target_path.exists() and not force:
            logger.info("Cached BSE Equities data found at %s. Skipping download.", target_path)
            return target_path

        logger.info("Starting download of BSE Equities (%d tickers)...", len(BSE_EQUITY_TICKERS))
        df = self._fetch_ticker_batch(BSE_EQUITY_TICKERS)
        if not df.empty:
            df.to_parquet(target_path, index=False, engine="pyarrow")
            logger.info("Saved BSE Equities to %s (%d records).", target_path, len(df))
        else:
            logger.error("Failed to produce dataset for BSE Equities.")
        return target_path

    def fetch_indices(self, force: bool = False) -> Path:
        """
        Fetches major market indices, saving to data/raw/indices.parquet.
        """
        target_path = self.raw_data_dir / "indices.parquet"
        if target_path.exists() and not force:
            logger.info("Cached Indices data found at %s. Skipping download.", target_path)
            return target_path

        logger.info("Starting download of Market Indices (%d tickers)...", len(INDEX_TICKERS))
        df = self._fetch_ticker_batch(INDEX_TICKERS)
        if not df.empty:
            df.to_parquet(target_path, index=False, engine="pyarrow")
            logger.info("Saved Indices to %s (%d records).", target_path, len(df))
        else:
            logger.error("Failed to produce dataset for Market Indices.")
        return target_path

    def fetch_macro(self, force: bool = False) -> Path:
        """
        Fetches commodities, currencies, and FRED economic series, saving to
        data/raw/macro.parquet.
        """
        target_path = self.raw_data_dir / "macro.parquet"
        if target_path.exists() and not force:
            logger.info("Cached Macro data found at %s. Skipping download.", target_path)
            return target_path

        logger.info("Starting download of Macro/Commodities and FRED Economic data...")
        collected: List[pd.DataFrame] = []

        # 1. Commodities & FX via yfinance
        for ticker in COMMODITY_FX_TICKERS:
            df = fetch_ticker_yfinance(ticker, self.start_date, self.end_date)
            if df is not None and not df.empty:
                collected.append(df)

        # 2. FRED Economic Series
        for series_id in FRED_SERIES_IDS:
            logger.info("Ingesting FRED series: %s", series_id)
            df = fetch_fred_series(series_id, self.start_date, self.end_date)
            if df is not None and not df.empty:
                collected.append(df)

        if collected:
            combined = pd.concat(collected, ignore_index=True)
            combined = combined.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)
            combined.to_parquet(target_path, index=False, engine="pyarrow")
            logger.info("Saved Macro & Economic data to %s (%d records).", target_path, len(combined))
        else:
            logger.error("Failed to produce dataset for Macro / Commodities.")
        return target_path

    def run_all(self, force: bool = False) -> Dict[str, Path]:
        """
        Executes ingestion for all defined market universes.
        """
        logger.info("=== Executing NiftyAlpha Data Ingestion Pipeline [force=%s] ===", force)
        results = {
            "nse_equities": self.fetch_nse_equities(force=force),
            "bse_equities": self.fetch_bse_equities(force=force),
            "indices": self.fetch_indices(force=force),
            "macro": self.fetch_macro(force=force),
        }
        logger.info("=== Data Ingestion Pipeline Execution Complete ===")
        return results


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NiftyAlpha Multi-Source Data Ingestion Module (Phase 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Ignore cached parquet files and force fresh downloads from all data sources.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2014-01-01",
        help="Historical start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (defaults to current system date).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_RAW_DATA_DIR),
        help="Target directory where .parquet files will be saved.",
    )
    parser.add_argument(
        "--universe",
        type=str,
        choices=["all", "nse", "bse", "indices", "macro"],
        default="all",
        help="Specify which data universe to ingest.",
    )

    args = parser.parse_args()

    pipeline = DataIngestionPipeline(
        raw_data_dir=Path(args.output_dir),
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if args.universe == "all":
        pipeline.run_all(force=args.force)
    elif args.universe == "nse":
        pipeline.fetch_nse_equities(force=args.force)
    elif args.universe == "bse":
        pipeline.fetch_bse_equities(force=args.force)
    elif args.universe == "indices":
        pipeline.fetch_indices(force=args.force)
    elif args.universe == "macro":
        pipeline.fetch_macro(force=args.force)


if __name__ == "__main__":
    main()
