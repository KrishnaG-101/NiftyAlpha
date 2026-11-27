"""
NiftyAlpha - Feature Engineering & Preprocessing Module
=======================================================
Production-grade quantitative feature engineering and temporal preprocessing.
Engineered to prevent lookahead bias and future data leakage through strict
time-series boundaries and partition isolation.

Pipelines & Features:
1. Alignment & Imputation:
   - Left-merge NSE Equities with BSE Equities, Indices, and Macro series on Date.
   - Forward-fill (ffill) macro and index series to bridge holiday differences.
2. Target Variable Creation (Per Ticker):
   - 20-trading-day forward return: (Close.shift(-20) - Close) / Close.
   - Binary growth indicator: (target_30d_return > 0).astype(int).
3. Technical Features (Per Ticker):
   - Momentum: RSI (14), MACD (12, 26, 9) [line, signal, histogram].
   - Volatility: ATR (14), Bollinger Bands (%B indicator).
   - Volume: On-Balance Volume (OBV), Volume / Volume.rolling(20).mean().
   - Volume Spread: Volume_NSE / Volume_BSE.
4. Cross-Market & Macro Features:
   - Market Breadth: 20-day rolling return of Sensex / Nifty ratio.
   - Macro Sensitivity: 30-day rolling correlation of stock returns with
     Brent Crude and USD/INR.
5. Calendar & Regime Encodings:
   - is_budget_month (February Union Budget indicator).
   - vix_regime_high (India VIX > 20).
   - Deterministic one-hot dummies for Month (1-12) and DayOfWeek (0-4).
6. Temporal Splitting & Scaling:
   - Train: Date <= 2021-12-31
   - Validation: 2022-01-01 <= Date <= 2023-12-31
   - Test: Date >= 2024-01-01
   - StandardScaler fitted STRICTLY on Train split; transforms Train, Val, Test.
   - Persists unscaled Date and Ticker metadata columns in all partitions.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NiftyAlpha.Features")

# Project Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

# Feature Schema Definitions
METADATA_COLUMNS: List[str] = ["Date", "Ticker"]

TARGET_COLUMNS: List[str] = [
    "target_30d_return",
    "is_positive_growth",
    "target_excess_return",
    "target_alpha_outperform",
]

NUMERIC_FEATURES: List[str] = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_14",
    "bollinger_pct_b",
    "obv",
    "volume_20d_ratio",
    "volume_nse_bse_ratio",
    "market_breadth_20d",
    "corr_brent_30d",
    "corr_usdinr_30d",
    "rel_strength_nifty_20d",
    "price_to_sma50",
    "brent_return_5d",
    "brent_return_20d",
    "usdinr_return_20d",
    "vix_change_5d",
]

CATEGORICAL_FEATURES: List[str] = [
    "is_budget_month",
    "vix_regime_high",
    "nifty_above_sma50",
] + [f"month_{m}" for m in range(1, 13)] + [f"day_{d}" for d in range(5)]


# ==============================================================================
# Technical Indicator Implementations (TA-Lib -> pandas-ta -> Native Vectorized)
# ==============================================================================

def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index with TA-Lib / pandas-ta / native fallback."""
    try:
        import talib
        res = talib.RSI(series.values.astype(float), timeperiod=period)
        return pd.Series(res, index=series.index)
    except Exception:
        pass

    try:
        import pandas_ta as ta
        res = ta.rsi(series, length=period)
        if res is not None and not res.empty:
            return res
    except Exception:
        pass

    # Native Wilder's smoothing implementation
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _calculate_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculates MACD, signal line, and histogram with multi-engine fallback."""
    try:
        import talib
        macd, macdsignal, macdhist = talib.MACD(
            series.values.astype(float), fastperiod=fast, slowperiod=slow, signalperiod=signal
        )
        return (
            pd.Series(macd, index=series.index),
            pd.Series(macdsignal, index=series.index),
            pd.Series(macdhist, index=series.index),
        )
    except Exception:
        pass

    try:
        import pandas_ta as ta
        res = ta.macd(series, fast=fast, slow=slow, signal=signal)
        if res is not None and not res.empty:
            return res.iloc[:, 0], res.iloc[:, 1], res.iloc[:, 2]
    except Exception:
        pass

    # Native EMA calculation
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Calculates Average True Range with multi-engine fallback."""
    try:
        import talib
        res = talib.ATR(
            high.values.astype(float),
            low.values.astype(float),
            close.values.astype(float),
            timeperiod=period,
        )
        return pd.Series(res, index=close.index)
    except Exception:
        pass

    try:
        import pandas_ta as ta
        res = ta.atr(high=high, low=low, close=close, length=period)
        if res is not None and not res.empty:
            return res
    except Exception:
        pass

    # Native True Range calculation
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def _calculate_bollinger_pct_b(
    series: pd.Series, period: int = 20, nbdev: float = 2.0
) -> pd.Series:
    """Calculates Bollinger Bands %B indicator: (Close - Lower) / (Upper - Lower)."""
    try:
        import talib
        upper, middle, lower = talib.BBANDS(
            series.values.astype(float), timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev
        )
        upper_s = pd.Series(upper, index=series.index)
        lower_s = pd.Series(lower, index=series.index)
        pct_b = (series - lower_s) / (upper_s - lower_s + 1e-9)
        return pct_b
    except Exception:
        pass

    try:
        import pandas_ta as ta
        bb = ta.bbands(series, length=period, std=nbdev)
        if bb is not None and not bb.empty:
            for col in bb.columns:
                if "BBP" in col.upper():
                    return bb[col]
    except Exception:
        pass

    # Native Rolling SMA and Standard Deviation
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std(ddof=0)
    upper = sma + (nbdev * std)
    lower = sma - (nbdev * std)
    pct_b = (series - lower) / (upper - lower + 1e-9)
    return pct_b


def _calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculates On-Balance Volume (OBV)."""
    try:
        import talib
        res = talib.OBV(close.values.astype(float), volume.values.astype(float))
        return pd.Series(res, index=close.index)
    except Exception:
        pass

    try:
        import pandas_ta as ta
        res = ta.obv(close=close, volume=volume)
        if res is not None and not res.empty:
            return res
    except Exception:
        pass

    # Native OBV
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    return obv


# ==============================================================================
# Feature Extraction Functions
# ==============================================================================

def compute_ticker_features(df_t: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical, macro-cross, and forward target variables strictly for
    a single stock ticker to prevent cross-sectional leakage.
    """
    df_t = df_t.sort_values(by="Date").copy()
    close = df_t["Close"]
    high = df_t["High"]
    low = df_t["Low"]
    volume = df_t["Volume"]

    # 1. Target Variables (20 trading days forward)
    fwd_ret = (close.shift(-20) - close) / close
    df_t["target_30d_return"] = fwd_ret
    df_t["is_positive_growth"] = (fwd_ret > 0.0).astype(int)

    if "Nifty_Close" in df_t.columns:
        nifty_fwd = (df_t["Nifty_Close"].shift(-20) - df_t["Nifty_Close"]) / (df_t["Nifty_Close"] + 1e-9)
        excess_ret = fwd_ret - nifty_fwd
        df_t["target_excess_return"] = excess_ret
        # Outperformance: Beats Nifty 50 forward return and has positive nominal growth
        df_t["target_alpha_outperform"] = ((excess_ret > 0.0) & (fwd_ret > 0.0)).astype(int)
    else:
        df_t["target_excess_return"] = fwd_ret
        df_t["target_alpha_outperform"] = (fwd_ret > 0.0).astype(int)

    # 2. Technical Features (Per Ticker)
    # Momentum
    df_t["rsi_14"] = _calculate_rsi(close, 14)
    macd, macd_signal, macd_hist = _calculate_macd(close, 12, 26, 9)
    df_t["macd"] = macd
    df_t["macd_signal"] = macd_signal
    df_t["macd_hist"] = macd_hist

    # Volatility
    df_t["atr_14"] = _calculate_atr(high, low, close, 14)
    df_t["bollinger_pct_b"] = _calculate_bollinger_pct_b(close, 20, 2.0)

    # Volume
    df_t["obv"] = _calculate_obv(close, volume)
    vol_mean_20 = volume.rolling(20).mean()
    df_t["volume_20d_ratio"] = volume / (vol_mean_20 + 1e-9)

    # BSE-NSE Volume Spread
    if "Volume_BSE" in df_t.columns:
        bse_vol = df_t["Volume_BSE"].replace(0.0, np.nan)
        df_t["volume_nse_bse_ratio"] = (volume / bse_vol).fillna(1.0).clip(0.0, 100.0)
    else:
        df_t["volume_nse_bse_ratio"] = 1.0

    # 3. Trend & Alpha Momentum Features
    # Price distance to 50-day SMA
    sma_50 = close.rolling(50).mean()
    df_t["price_to_sma50"] = ((close - sma_50) / (sma_50 + 1e-9)).fillna(0.0)

    # Relative Strength to Nifty 50 (20-day return spread)
    if "Nifty_Close" in df_t.columns:
        stock_20d = close.pct_change(20)
        nifty_20d = df_t["Nifty_Close"].pct_change(20)
        df_t["rel_strength_nifty_20d"] = (stock_20d - nifty_20d).fillna(0.0)
    else:
        df_t["rel_strength_nifty_20d"] = 0.0

    # 4. Cross-Market & Macro Features
    if "Sensex_Close" in df_t.columns and "Nifty_Close" in df_t.columns:
        sensex_nifty_ratio = df_t["Sensex_Close"] / (df_t["Nifty_Close"] + 1e-9)
        ratio_shift_20 = sensex_nifty_ratio.shift(20)
        df_t["market_breadth_20d"] = (sensex_nifty_ratio - ratio_shift_20) / (ratio_shift_20 + 1e-9)
    else:
        df_t["market_breadth_20d"] = 0.0

    stock_ret = close.pct_change()

    # Macro 30-day Rolling Correlations & Velocities
    if "Brent_Close" in df_t.columns:
        brent_ret = df_t["Brent_Close"].pct_change()
        df_t["corr_brent_30d"] = stock_ret.rolling(30).corr(brent_ret).fillna(0.0)
        df_t["brent_return_5d"] = df_t["Brent_Close"].pct_change(5).fillna(0.0)
        df_t["brent_return_20d"] = df_t["Brent_Close"].pct_change(20).fillna(0.0)
    else:
        df_t["corr_brent_30d"] = 0.0
        df_t["brent_return_5d"] = 0.0
        df_t["brent_return_20d"] = 0.0

    if "USDINR_Close" in df_t.columns:
        usdinr_ret = df_t["USDINR_Close"].pct_change()
        df_t["corr_usdinr_30d"] = stock_ret.rolling(30).corr(usdinr_ret).fillna(0.0)
        df_t["usdinr_return_20d"] = df_t["USDINR_Close"].pct_change(20).fillna(0.0)
    else:
        df_t["corr_usdinr_30d"] = 0.0
        df_t["usdinr_return_20d"] = 0.0

    if "India_VIX" in df_t.columns:
        df_t["vix_change_5d"] = df_t["India_VIX"].diff(5).fillna(0.0)
    else:
        df_t["vix_change_5d"] = 0.0

    return df_t


def add_categorical_and_dummy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-based and market regime indicators, generating fixed deterministic
    one-hot dummies to guarantee column alignment across temporal partitions.
    """
    df = df.copy()
    dates = pd.to_datetime(df["Date"])
    month = dates.dt.month
    day_of_week = dates.dt.day_of_week if hasattr(dates.dt, "day_of_week") else dates.dt.dayofweek

    # Indian Union Budget Indicator (February)
    df["is_budget_month"] = (month == 2).astype(int)

    # India VIX Volatility Regime
    if "India_VIX" in df.columns:
        df["vix_regime_high"] = (df["India_VIX"] > 20.0).astype(int)
    else:
        df["vix_regime_high"] = 0

    # Nifty 50 Trend Regime (Above 50-day SMA)
    if "Nifty_Close" in df.columns:
        nifty_sma50 = df["Nifty_Close"].rolling(50).mean()
        df["nifty_above_sma50"] = (df["Nifty_Close"] > nifty_sma50).fillna(1).astype(int)
    else:
        df["nifty_above_sma50"] = 1

    # Deterministic Month Dummies (1 to 12)
    for m in range(1, 13):
        df[f"month_{m}"] = (month == m).astype(int)

    # Deterministic Day of Week Dummies (0: Mon, 1: Tue, 2: Wed, 3: Thu, 4: Fri)
    for d in range(5):
        df[f"day_{d}"] = (day_of_week == d).astype(int)

    return df


# ==============================================================================
# Pipeline Orchestrator
# ==============================================================================

class FeatureEngineeringPipeline:
    """
    Coordinates data loading, multi-universe alignment, technical feature computation,
    leakage-free temporal splitting, and scaling.
    """

    def __init__(
        self,
        raw_dir: Path = DEFAULT_RAW_DIR,
        processed_dir: Path = DEFAULT_PROCESSED_DIR,
        models_dir: Path = DEFAULT_MODELS_DIR,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)

    def load_and_align_datasets(self) -> pd.DataFrame:
        """
        Loads raw parquet datasets and performs alignment on Date.
        Forward-fills macro and index series to avoid holiday misalignment.
        """
        nse_path = self.raw_dir / "nse_equities.parquet"
        bse_path = self.raw_dir / "bse_equities.parquet"
        indices_path = self.raw_dir / "indices.parquet"
        macro_path = self.raw_dir / "macro.parquet"

        for p in [nse_path, bse_path, indices_path, macro_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Required raw dataset missing at {p}. "
                    f"Please run 'make fetch-data' or 'python src/fetch_data.py' first."
                )

        logger.info("Loading raw parquet files from %s...", self.raw_dir)
        nse_df = pd.read_parquet(nse_path)
        bse_df = pd.read_parquet(bse_path)
        indices_df = pd.read_parquet(indices_path)
        macro_df = pd.read_parquet(macro_path)

        # Standardize dates
        for df in [nse_df, bse_df, indices_df, macro_df]:
            df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

        # 1. Pivot Indices on Date
        idx_pivot = indices_df.pivot(index="Date", columns="Ticker", values="Close")
        idx_rename = {"^NSEI": "Nifty_Close", "^BSESN": "Sensex_Close", "^INDIAVIX": "India_VIX"}
        idx_pivot = idx_pivot.rename(columns=idx_rename)
        idx_pivot = idx_pivot.ffill().bfill()

        # 2. Pivot Macro on Date
        macro_pivot = macro_df.pivot(index="Date", columns="Ticker", values="Close")
        macro_rename = {
            "BZ=F": "Brent_Close",
            "GC=F": "Gold_Close",
            "INR=X": "USDINR_Close",
            "DGS10": "US10Y_Yield",
            "FEDFUNDS": "FedFunds_Rate",
        }
        macro_pivot = macro_pivot.rename(columns=macro_rename)
        macro_pivot = macro_pivot.ffill().bfill()

        # Combine Macro and Indices context
        context_df = pd.merge(idx_pivot, macro_pivot, on="Date", how="outer").sort_index()
        context_df = context_df.ffill().bfill()

        # 3. Align BSE Volume with NSE Equities (map .BO -> .NS)
        bse_mapped = bse_df.copy()
        bse_mapped["Ticker"] = bse_mapped["Ticker"].str.replace(".BO", ".NS")
        bse_sub = bse_mapped[["Date", "Ticker", "Volume"]].rename(columns={"Volume": "Volume_BSE"})

        # Left-merge onto NSE Equities
        aligned_df = pd.merge(nse_df, bse_sub, on=["Date", "Ticker"], how="left")
        aligned_df = pd.merge(aligned_df, context_df, on="Date", how="left")

        # Forward-fill any residual missing context across dates
        for col in context_df.columns:
            aligned_df[col] = aligned_df[col].ffill().bfill()

        logger.info("Datasets aligned successfully. Total records: %d", len(aligned_df))
        return aligned_df

    def process_features(self, aligned_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts per-ticker technical features, cross-market features,
        and global calendar encodings.
        """
        logger.info("Computing per-ticker technical features and forward returns...")
        ticker_dfs: List[pd.DataFrame] = []
        unique_tickers = aligned_df["Ticker"].unique()

        for idx, ticker in enumerate(unique_tickers, start=1):
            group = aligned_df[aligned_df["Ticker"] == ticker].copy()
            featured_group = compute_ticker_features(group)
            ticker_dfs.append(featured_group)

        full_df = pd.concat(ticker_dfs, ignore_index=True)

        logger.info("Adding calendar, budget, and regime encodings...")
        full_df = add_categorical_and_dummy_features(full_df)

        return full_df

    def temporal_split_and_scale(self, df: pd.DataFrame) -> Dict[str, Path]:
        """
        Executes strict temporal splitting and scaler fitting exclusively on the Train split.
        Saves train, validation, and test partitions to data/processed/.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Drop NaN rows resulting from rolling windows (e.g., 30d corr) and shift(-20)
        initial_len = len(df)
        df_clean = df.dropna(subset=NUMERIC_FEATURES + TARGET_COLUMNS).copy()
        dropped = initial_len - len(df_clean)
        logger.info(
            "Dropped NaN boundary rows: %d -> %d (removed %d burn-in and forward shift rows)",
            initial_len,
            len(df_clean),
            dropped,
        )

        df_clean = df_clean.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)

        # Strict Temporal Masks
        train_mask = df_clean["Date"] <= pd.Timestamp("2021-12-31")
        val_mask = (df_clean["Date"] >= pd.Timestamp("2022-01-01")) & (
            df_clean["Date"] <= pd.Timestamp("2023-12-31")
        )
        test_mask = df_clean["Date"] >= pd.Timestamp("2024-01-01")

        train_df = df_clean[train_mask].copy()
        val_df = df_clean[val_mask].copy()
        test_df = df_clean[test_mask].copy()

        logger.info(
            "Temporal split completed -> Train: %d rows | Val: %d rows | Test: %d rows",
            len(train_df),
            len(val_df),
            len(test_df),
        )

        if len(train_df) == 0:
            raise ValueError("Train split is empty. Verify date range of source dataset.")

        # Standardize Continuous Numeric Features (FIT ONLY ON TRAIN)
        scaler = StandardScaler()
        scaler.fit(train_df[NUMERIC_FEATURES])
        logger.info(
            "Fitted StandardScaler strictly on Train split (%d rows, %d numeric features).",
            len(train_df),
            len(NUMERIC_FEATURES),
        )

        train_df[NUMERIC_FEATURES] = scaler.transform(train_df[NUMERIC_FEATURES])
        if len(val_df) > 0:
            val_df[NUMERIC_FEATURES] = scaler.transform(val_df[NUMERIC_FEATURES])
        if len(test_df) > 0:
            test_df[NUMERIC_FEATURES] = scaler.transform(test_df[NUMERIC_FEATURES])

        # Persist Scaler artifact for inference
        scaler_path = self.models_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        logger.info("Saved fitted scaler artifact to %s", scaler_path)

        # Assemble partition schemas preserving unscaled Date and Ticker metadata
        feature_cols = METADATA_COLUMNS + NUMERIC_FEATURES + CATEGORICAL_FEATURES
        target_cols = METADATA_COLUMNS + TARGET_COLUMNS

        X_train = train_df[feature_cols]
        y_train = train_df[target_cols]
        X_val = val_df[feature_cols]
        y_val = val_df[target_cols]
        X_test = test_df[feature_cols]
        y_test = test_df[target_cols]

        outputs = {
            "X_train": self.processed_dir / "X_train.parquet",
            "y_train": self.processed_dir / "y_train.parquet",
            "X_val": self.processed_dir / "X_val.parquet",
            "y_val": self.processed_dir / "y_val.parquet",
            "X_test": self.processed_dir / "X_test.parquet",
            "y_test": self.processed_dir / "y_test.parquet",
        }

        X_train.to_parquet(outputs["X_train"], index=False, engine="pyarrow")
        y_train.to_parquet(outputs["y_train"], index=False, engine="pyarrow")
        X_val.to_parquet(outputs["X_val"], index=False, engine="pyarrow")
        y_val.to_parquet(outputs["y_val"], index=False, engine="pyarrow")
        X_test.to_parquet(outputs["X_test"], index=False, engine="pyarrow")
        y_test.to_parquet(outputs["y_test"], index=False, engine="pyarrow")

        for name, path in outputs.items():
            logger.info("Persisted %s to %s", name, path)

        return outputs

    def run(self) -> Dict[str, Path]:
        """Executes the end-to-end feature pipeline."""
        logger.info("=== Starting NiftyAlpha Feature Engineering Pipeline ===")
        aligned = self.load_and_align_datasets()
        featured = self.process_features(aligned)
        results = self.temporal_split_and_scale(featured)
        logger.info("=== Feature Engineering Pipeline Complete ===")
        return results


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NiftyAlpha Feature Engineering & Preprocessing Module",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="Directory containing raw parquet datasets.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR),
        help="Directory where processed train/val/test parquets will be stored.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help="Directory where scaler artifacts will be saved.",
    )

    args = parser.parse_args()

    pipeline = FeatureEngineeringPipeline(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        models_dir=Path(args.models_dir),
    )
    pipeline.run()


if __name__ == "__main__":
    main()
