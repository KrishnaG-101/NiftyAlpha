#!/usr/bin/env python3
"""
NiftyAlpha - Automated Daily Predictor & Order Generator
========================================================
Automated end-of-day inference and order generation pipeline designed to be
scheduled via Linux cron (crontab -e) or Windows Task Scheduler.

Execution Workflow:
1. Slices the latest available market date from the feature-engineered dataset.
2. Loads the trained predictive model (e.g. XGBoost) from serialized models.
3. Computes probability scores across all 50 constituent Nifty equities.
4. Generates structured trade directives with deterministic risk controls:
   - Action: BUY (if P >= threshold) vs. HOLD / CASH
   - Execution: Market Open (Day t+1)
   - Dynamic Stop-Loss (-8% barrier in INR)
   - Dynamic Take-Profit (+15% target in INR)
   - Time Horizon Exit (20 Trading Days)
5. Persists actionable order manifests to:
   - reports/daily_orders_<YYYY-MM-DD>.csv
   - reports/daily_orders_<YYYY-MM-DD>.json
   - reports/latest_daily_orders.csv (symlinked/latest copy)
   - reports/latest_daily_orders.json
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Configure logging with timestamps for cron logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NiftyAlpha.CronPredictor")

# Project Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


class DailyPredictor:
    """
    Automated Daily Predictor for scheduled cron or task scheduler execution.
    """

    def __init__(
        self,
        processed_dir: Path = DEFAULT_PROCESSED_DIR,
        raw_dir: Path = DEFAULT_RAW_DIR,
        models_dir: Path = DEFAULT_MODELS_DIR,
        reports_dir: Path = DEFAULT_REPORTS_DIR,
        model_name: str = "xgboost",
        threshold: float = 0.50,
    ):
        self.processed_dir = Path(processed_dir)
        self.raw_dir = Path(raw_dir)
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.model_name = model_name.lower().strip()
        self.threshold = threshold

    def load_model(self) -> Tuple[Any, str]:
        """Loads the serialized model artifact."""
        model_file = self.models_dir / f"{self.model_name}_model.joblib"
        if model_file.exists():
            logger.info("Loading model artifact from %s", model_file)
            return joblib.load(model_file), self.model_name

        # Fallback to any available trained model
        available_models = list(self.models_dir.glob("*_model.joblib"))
        if not available_models:
            raise FileNotFoundError(
                f"No serialized model artifacts found in {self.models_dir}. "
                "Please run 'python main.py train' first."
            )
        fallback = available_models[0]
        loaded_name = fallback.stem.replace("_model", "")
        logger.warning(
            "Model '%s' not found. Falling back to '%s' at %s",
            self.model_name,
            loaded_name,
            fallback,
        )
        return joblib.load(fallback), loaded_name

    def run(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes daily inference and persists dated and latest order manifests.

        Parameters:
        -----------
        target_date : Optional[str]
            Specific market date ('YYYY-MM-DD'). If None, isolates the most
            recent available date in the dataset.

        Returns:
        --------
        summary : Dict[str, Any]
            Execution metadata, signal counts, and export paths.
        """
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load feature matrix (prefer X_test.parquet, fallback to X_val or X_train)
        feat_path = self.processed_dir / "X_test.parquet"
        if not feat_path.exists():
            feat_path = self.processed_dir / "X_val.parquet"
        if not feat_path.exists():
            raise FileNotFoundError(
                f"Processed feature matrix not found in {self.processed_dir}. "
                "Please run 'python main.py features' first."
            )

        logger.info("Reading feature matrix from %s...", feat_path)
        X_df = pd.read_parquet(feat_path)
        X_df["Date"] = pd.to_datetime(X_df["Date"])

        # 2. Load raw price series for closing price mapping
        raw_nse_path = self.raw_dir / "nse_equities.parquet"
        raw_prices: Optional[pd.DataFrame] = None
        if raw_nse_path.exists():
            logger.info("Reading raw market data from %s...", raw_nse_path)
            raw_prices = pd.read_parquet(raw_nse_path)
            raw_prices["Date"] = pd.to_datetime(raw_prices["Date"])

        # 3. Determine target evaluation date
        if target_date is not None:
            eval_dt = pd.to_datetime(target_date)
            day_features = X_df[X_df["Date"] == eval_dt].copy()
            if day_features.empty:
                raise ValueError(
                    f"No feature records found for requested date {target_date}."
                )
        else:
            eval_dt = X_df["Date"].max()
            day_features = X_df[X_df["Date"] == eval_dt].copy()

        date_str = eval_dt.strftime("%Y-%m-%d")
        next_bday = (eval_dt + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
        logger.info(
            "Screening universe on market close date: %s (Execution Date: %s)",
            date_str,
            next_bday,
        )

        # 4. Load Model & Generate Inference Probabilities
        model, active_model_name = self.load_model()
        meta_cols = ["Date", "Ticker"]
        feature_cols = [c for c in day_features.columns if c not in meta_cols]
        X_feats = day_features[feature_cols].copy()

        if hasattr(model, "predict_proba"):
            probs_raw = model.predict_proba(X_feats)
            probs = (
                probs_raw[:, 1]
                if probs_raw.ndim == 2 and probs_raw.shape[1] > 1
                else np.ravel(probs_raw)
            )
        else:
            probs = model.predict(X_feats)

        day_features["confidence_score"] = np.round(probs, 4)
        day_features["is_buy_signal"] = day_features["confidence_score"] >= self.threshold

        # 5. Attach closing prices
        if raw_prices is not None:
            raw_day = raw_prices[raw_prices["Date"] == eval_dt][["Ticker", "Close"]].rename(
                columns={"Close": "last_close_inr"}
            )
            merged = pd.merge(day_features, raw_day, on="Ticker", how="left")
        else:
            merged = day_features.copy()
            merged["last_close_inr"] = np.nan

        # 6. Format deterministic order directives
        orders: List[Dict[str, Any]] = []
        for _, row in merged.iterrows():
            ticker = str(row["Ticker"])
            conf = float(row["confidence_score"])
            is_buy = bool(row["is_buy_signal"])
            close_px = (
                float(row["last_close_inr"]) if pd.notna(row["last_close_inr"]) else None
            )

            action = "BUY" if is_buy else "HOLD"
            pos_size = 30.0 if is_buy else 0.0

            stop_loss = round(close_px * 0.92, 2) if close_px is not None else None
            take_profit = round(close_px * 1.15, 2) if close_px is not None else None

            order_entry = {
                "signal_date": date_str,
                "execution_date": next_bday,
                "ticker": ticker,
                "action": action,
                "model_name": active_model_name,
                "confidence_score": conf,
                "confidence_pct": f"{conf * 100:.2f}%",
                "threshold_applied": self.threshold,
                "last_close_inr": close_px,
                "execution_directive": "ENTER LONG (Market Open t+1)" if is_buy else "STAY DEFENSIVE / CASH",
                "target_allocation_pct": pos_size,
                "stop_loss_inr": stop_loss,
                "take_profit_inr": take_profit,
                "max_holding_trading_days": 20,
            }
            orders.append(order_entry)

        orders_df = pd.DataFrame(orders)
        orders_df = orders_df.sort_values(by="confidence_score", ascending=False).reset_index(drop=True)

        # 7. Persist daily dated reports and latest copies
        dated_csv_path = self.reports_dir / f"daily_orders_{date_str}.csv"
        dated_json_path = self.reports_dir / f"daily_orders_{date_str}.json"
        latest_csv_path = self.reports_dir / "latest_daily_orders.csv"
        latest_json_path = self.reports_dir / "latest_daily_orders.json"

        orders_df.to_csv(dated_csv_path, index=False)
        orders_df.to_csv(latest_csv_path, index=False)

        # JSON serialization
        json_payload = {
            "metadata": {
                "signal_date": date_str,
                "execution_date": next_bday,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_name": active_model_name,
                "threshold": self.threshold,
                "universe_count": len(orders_df),
                "buy_orders_count": int(np.sum(orders_df["action"] == "BUY")),
            },
            "orders": orders_df.to_dict(orient="records"),
        }

        with open(dated_json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2)
        with open(latest_json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2)

        buy_count = int(np.sum(orders_df["action"] == "BUY"))
        logger.info(
            "=== DAILY CRON ORDERS GENERATED [%s] ===", date_str
        )
        logger.info(
            "Universe Screened: %d Equities | Buy Signals: %d (Threshold: %.2f)",
            len(orders_df),
            buy_count,
            self.threshold,
        )
        logger.info("Saved dated orders to %s and %s", dated_csv_path, dated_json_path)
        logger.info("Updated latest orders at %s and %s", latest_csv_path, latest_json_path)

        # Log Top Recommendations
        top_buys = orders_df[orders_df["action"] == "BUY"].head(5)
        if not top_buys.empty:
            logger.info("Top Buy Directives:")
            for _, r in top_buys.iterrows():
                px_str = f"₹{r['last_close_inr']:,.2f}" if r["last_close_inr"] else "N/A"
                sl_str = f"₹{r['stop_loss_inr']:,.2f}" if r["stop_loss_inr"] else "N/A"
                tp_str = f"₹{r['take_profit_inr']:,.2f}" if r["take_profit_inr"] else "N/A"
                logger.info(
                    "  [BUY] %-12s | Conf: %s | Close: %s | SL: %s | TP: %s",
                    r["ticker"],
                    r["confidence_pct"],
                    px_str,
                    sl_str,
                    tp_str,
                )
        else:
            logger.info("No equities exceeded the %.2f confidence cutoff. Portfolio posture: 100%% Cash.", self.threshold)

        return {
            "date": date_str,
            "execution_date": next_bday,
            "universe_count": len(orders_df),
            "buy_count": buy_count,
            "dated_csv": str(dated_csv_path),
            "dated_json": str(dated_json_path),
            "latest_csv": str(latest_csv_path),
            "latest_json": str(latest_json_path),
            "orders": orders_df,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NiftyAlpha Automated Daily Predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR),
        help="Directory containing processed feature parquet files.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="Directory containing raw market parquet files.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help="Directory containing trained model artifacts (.joblib).",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory where daily order manifests will be persisted.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        help="Model architecture to use for inference.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Model probability threshold for generating BUY orders.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Specific date to evaluate ('YYYY-MM-DD'). Defaults to latest available market date.",
    )

    args = parser.parse_args()

    predictor = DailyPredictor(
        processed_dir=Path(args.processed_dir),
        raw_dir=Path(args.raw_dir),
        models_dir=Path(args.models_dir),
        reports_dir=Path(args.reports_dir),
        model_name=args.model,
        threshold=args.threshold,
    )
    predictor.run(target_date=args.date)


if __name__ == "__main__":
    main()
