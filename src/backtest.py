"""
NiftyAlpha - Realistic Trading Simulation Module (Phase 5)
==========================================================
Day-by-day iterative backtesting engine on out-of-sample data (2024-present).
Simulates real-world market microstructure, trade friction, risk management,
and portfolio mark-to-market accounting.

Key Simulation Rules:
- Starting Capital: INR 1,000,000.
- Portfolio Diversification: Max 5 concurrent positions, 20% equal weighting.
- Transaction Friction: 0.10% entry + 0.10% exit (0.20% round-trip) covering
  Indian STT, exchange fees, brokerage, and slippage.
- Chronological Order of Operations:
    1. Market Open: Evaluate Stop Loss (-8%), Take Profit (+15%), Time Exit (20 days).
    2. Signal Execution: Execute yesterday's Buy Signals at today's Open price.
    3. End of Day: Mark-to-market active positions using Close price, evaluate today's signals.
- Benchmark Comparison: Compares performance against Buy & Hold on Nifty 50 (^NSEI).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NiftyAlpha.Backtest")

# Project Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

# Simulation Constants
DEFAULT_CAPITAL = 1_000_000.0
DEFAULT_MAX_POSITIONS = 5
POSITION_ALLOCATION = 0.20
TRANSACTION_FRICTION = 0.0010  # 0.10% per transaction leg
STOP_LOSS_PCT = 0.08           # -8% Stop Loss
TAKE_PROFIT_PCT = 0.15         # +15% Take Profit
MAX_HOLDING_DAYS = 20          # 20 trading days exit
RISK_FREE_RATE = 0.07          # 7% Indian Sovereign Risk-Free Rate
DEFAULT_PROB_THRESHOLD = 0.60


# ==============================================================================
# Model & Data Loaders
# ==============================================================================

def load_best_model(models_dir: Path, preferred_name: Optional[str] = None) -> Tuple[Any, str]:
    """
    Selects and loads the best trained model artifact from the models directory.
    Checks reports/metrics.json, prioritizing calibrated ensemble architectures
    (xgboost -> random_forest -> logistic_regression) over single decision trees.
    """
    metrics_file = models_dir.parent / "reports" / "metrics.json"

    if preferred_name:
        candidate_path = models_dir / f"{preferred_name}_model.joblib"
        if candidate_path.exists():
            logger.info("Loading specified model artifact: %s", candidate_path)
            return joblib.load(candidate_path), preferred_name

    # Check metrics.json if available to find best validation PR-AUC / ROC-AUC
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            if metrics:
                # Prioritize ensemble models for out-of-sample stability
                ensemble_keys = [k for k in ["xgboost", "random_forest", "logistic_regression"] if k in metrics]
                candidate_pool = {k: metrics[k] for k in ensemble_keys} if ensemble_keys else metrics
                best_name = max(candidate_pool.keys(), key=lambda k: candidate_pool[k].get("pr_auc", 0.0))
                best_path = models_dir / f"{best_name}_model.joblib"
                if best_path.exists():
                    logger.info("Auto-selected best model from metrics.json: %s (PR-AUC: %.4f)",
                                best_name, candidate_pool[best_name].get("pr_auc", 0.0))
                    return joblib.load(best_path), best_name
        except Exception as e:
            logger.warning("Could not parse metrics.json for auto-selection: %s", e)

    # Fallback search order
    priority_order = [
        "xgboost_model.joblib",
        "random_forest_model.joblib",
        "logistic_regression_model.joblib",
        "mlp_neural_network_model.joblib",
        "decision_tree_model.joblib",
    ]

    for filename in priority_order:
        p = models_dir / filename
        if p.exists():
            model_name = filename.replace("_model.joblib", "")
            logger.info("Loaded available model: %s", p)
            return joblib.load(p), model_name

    all_models = list(models_dir.glob("*.joblib"))
    if all_models:
        first = all_models[0]
        return joblib.load(first), first.stem.replace("_model", "")

    raise FileNotFoundError(
        f"No trained model artifacts found in {models_dir}. "
        f"Please run 'make train' or 'python src/models.py' first."
    )


def load_simulation_data(
    processed_dir: Path,
    raw_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads test features, targets, unscaled raw price data, and benchmark indices.
    """
    x_test_path = processed_dir / "X_test.parquet"
    y_test_path = processed_dir / "y_test.parquet"
    raw_nse_path = raw_dir / "nse_equities.parquet"
    indices_path = raw_dir / "indices.parquet"

    for p in [x_test_path, y_test_path, raw_nse_path, indices_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required data file missing at {p}. "
                f"Verify preprocessing and raw data ingestion steps."
            )

    logger.info("Loading test partitions and raw market datasets...")
    X_test_df = pd.read_parquet(x_test_path)
    y_test_df = pd.read_parquet(y_test_path)
    raw_nse_df = pd.read_parquet(raw_nse_path)
    indices_df = pd.read_parquet(indices_path)

    # Standardize dates
    for df in [X_test_df, y_test_df, raw_nse_df, indices_df]:
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    return X_test_df, y_test_df, raw_nse_df, indices_df


# ==============================================================================
# Performance Analytics
# ==============================================================================

def calculate_performance_metrics(
    equity_series: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Dict[str, float]:
    """
    Calculates quantitative metrics: Total Return, CAGR, Max Drawdown,
    and Annualized Sharpe Ratio based on a 252 trading day convention.
    """
    if len(equity_series) < 2:
        return {
            "Total Return (%)": 0.0,
            "CAGR (%)": 0.0,
            "Max Drawdown (%)": 0.0,
            "Annualized Sharpe": 0.0,
        }

    initial_val = equity_series.iloc[0]
    final_val = equity_series.iloc[-1]
    n_days = len(equity_series)

    total_return = (final_val - initial_val) / initial_val
    num_years = max(n_days / 252.0, 0.01)
    cagr = (final_val / initial_val) ** (1.0 / num_years) - 1.0

    # Drawdown
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_drawdown = float(drawdown.min())

    # Daily returns & Sharpe Ratio
    daily_returns = equity_series.pct_change().dropna()
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    excess_returns = daily_returns - rf_daily

    if daily_returns.std() > 1e-9:
        sharpe_ratio = float(np.sqrt(252.0) * (excess_returns.mean() / daily_returns.std()))
    else:
        sharpe_ratio = 0.0

    return {
        "Total Return (%)": float(total_return * 100.0),
        "CAGR (%)": float(cagr * 100.0),
        "Max Drawdown (%)": float(max_drawdown * 100.0),
        "Annualized Sharpe": float(sharpe_ratio),
    }


# ==============================================================================
# Simulation Engine
# ==============================================================================

class TradingSimulator:
    """
    Executes a realistic day-by-day trading simulation with microstructure constraints,
    friction deductions, and risk limits.
    """

    def __init__(
        self,
        initial_capital: float = DEFAULT_CAPITAL,
        max_positions: int = DEFAULT_MAX_POSITIONS,
        position_alloc: float = POSITION_ALLOCATION,
        friction: float = TRANSACTION_FRICTION,
        stop_loss_pct: float = STOP_LOSS_PCT,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        max_holding_days: int = MAX_HOLDING_DAYS,
        prob_threshold: float = DEFAULT_PROB_THRESHOLD,
    ):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_alloc = position_alloc
        self.friction = friction
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_days = max_holding_days
        self.prob_threshold = prob_threshold

    def run_simulation(
        self,
        X_test: pd.DataFrame,
        raw_prices: pd.DataFrame,
        indices_df: pd.DataFrame,
        model: Any,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Runs the chronological day-by-day event loop.
        """
        logger.info(
            f"Initializing simulation engine [Initial Capital: INR {self.initial_capital:,.2f}, Threshold: {self.prob_threshold:.2f}]..."
        )

        # 1. Generate Model Predictions
        meta_cols = ["Date", "Ticker"]
        feature_cols = [c for c in X_test.columns if c not in meta_cols]
        X_feats = X_test[feature_cols]

        if hasattr(model, "predict_proba"):
            raw_probs = model.predict_proba(X_feats)
            if hasattr(raw_probs, "ndim") and raw_probs.ndim == 2 and raw_probs.shape[1] > 1:
                probs = raw_probs[:, 1]
            else:
                probs = np.ravel(raw_probs)
        else:
            probs = model.predict(X_feats)

        signal_df = X_test[meta_cols].copy()
        signal_df["Prob"] = probs
        signal_df["Buy_Signal"] = signal_df["Prob"] >= self.prob_threshold
        if "rsi_14" in X_test.columns:
            signal_df["rsi_14"] = X_test["rsi_14"]
        if "price_to_sma50" in X_test.columns:
            signal_df["price_to_sma50"] = X_test["price_to_sma50"]

        logger.info("Generated predictions across %d samples. Positive signals (Buy): %d",
                    len(signal_df), int(signal_df["Buy_Signal"].sum()))

        # Pre-group signals by date for high-speed simulation loop
        signals_by_date = {d: group for d, group in signal_df.groupby("Date")}

        # 2. Build Fast Price Lookup Table: (Date, Ticker) -> Price Row
        raw_lookup = {}
        for row in raw_prices.itertuples(index=False):
            raw_lookup[(row.Date, row.Ticker)] = row

        # 3. Build Benchmark Equity Curve (^NSEI) and VIX regime map
        nifty_bars = indices_df[indices_df["Ticker"] == "^NSEI"].sort_values(by="Date").copy()
        vix_bars = indices_df[indices_df["Ticker"] == "^INDIAVIX"].sort_values(by="Date").copy()
        vix_lookup = {r.Date: r.Close for r in vix_bars.itertuples(index=False)}

        test_dates = sorted(signal_df["Date"].unique())

        if not test_dates:
            raise ValueError("Test dataset contains no unique dates.")

        first_test_date = test_dates[0]
        nifty_test = nifty_bars[nifty_bars["Date"] >= first_test_date].copy()

        if not nifty_test.empty:
            base_nifty_close = nifty_test.iloc[0]["Close"]
            nifty_lookup = {
                r.Date: (r.Close / base_nifty_close) * self.initial_capital
                for r in nifty_test.itertuples(index=False)
            }
        else:
            nifty_lookup = {}

        # 4. Simulation State Variables
        cash = self.initial_capital
        active_positions: Dict[str, Dict[str, Any]] = {}
        trade_log: List[Dict[str, Any]] = []
        equity_curve: List[Dict[str, Any]] = []

        yesterday_signals: List[Dict[str, Any]] = []

        # Chronological day-by-day simulation loop
        for current_date in test_dates:
            # ------------------------------------------------------------------
            # Step A: Market Open - Manage Existing Active Positions
            # ------------------------------------------------------------------
            exited_tickers = []
            for ticker, pos in active_positions.items():
                pos["days_held"] += 1
                price_bar = raw_lookup.get((current_date, ticker))

                if price_bar is None:
                    # Missing bar on this date (e.g., stock suspended / holiday)
                    continue

                low_p = float(price_bar.Low)
                high_p = float(price_bar.High)
                open_p = float(price_bar.Open)
                entry_p = float(pos["entry_price"])

                # Record peak gain
                peak_gain = (high_p - entry_p) / entry_p
                if peak_gain > pos.get("max_gain", 0.0):
                    pos["max_gain"] = peak_gain

                exit_triggered = False
                exit_price = 0.0
                exit_reason = ""

                # Condition 0: Dynamic Profit Protection (Trailing Breakeven Stop)
                # If stock reached >= 4.5% peak gain, trail stop to breakeven (+0.2% to cover friction)
                if pos.get("max_gain", 0.0) >= 0.045 and low_p <= entry_p * 1.002:
                    exit_triggered = True
                    exit_reason = "Trailing Breakeven"
                    exit_price = min(open_p, entry_p * 1.002)

                # Condition 1: Stop Loss (-8%)
                elif low_p <= entry_p * (1.0 - self.stop_loss_pct):
                    exit_triggered = True
                    exit_reason = "Stop Loss"
                    # Exit at stop loss or open gap down
                    exit_price = min(open_p, entry_p * (1.0 - self.stop_loss_pct))

                # Condition 2: Take Profit (+15%)
                elif high_p >= entry_p * (1.0 + self.take_profit_pct):
                    exit_triggered = True
                    exit_reason = "Take Profit"
                    # Exit at take profit or open gap up
                    exit_price = max(open_p, entry_p * (1.0 + self.take_profit_pct))

                # Condition 3: Time Limit (20 trading days)
                elif pos["days_held"] >= self.max_holding_days:
                    exit_triggered = True
                    exit_reason = "Time Limit"
                    exit_price = open_p

                if exit_triggered:
                    gross_exit = pos["shares"] * exit_price
                    exit_friction = gross_exit * self.friction
                    net_exit = gross_exit - exit_friction
                    cash += net_exit

                    cost_basis = pos["shares"] * entry_p * (1.0 + self.friction)
                    pnl_pct = (net_exit / cost_basis) - 1.0

                    trade_log.append({
                        "Ticker": ticker,
                        "Entry_Date": pos["entry_date"].strftime("%Y-%m-%d"),
                        "Entry_Price": round(entry_p, 2),
                        "Exit_Date": current_date.strftime("%Y-%m-%d"),
                        "Exit_Price": round(exit_price, 2),
                        "Return_Pct": round(pnl_pct * 100.0, 2),
                        "Exit_Reason": exit_reason,
                    })
                    exited_tickers.append(ticker)

            for t_ex in exited_tickers:
                del active_positions[t_ex]

            # ------------------------------------------------------------------
            # Step B: Signal Processing - Execute Yesterday's Buy Signals
            # ------------------------------------------------------------------
            # Macro Regime Gate: Prohibit opening new long positions when India VIX > 22.0
            vix_val = vix_lookup.get(current_date, 15.0)
            allow_regime_entries = (vix_val < 22.0)

            if yesterday_signals and len(active_positions) < self.max_positions and allow_regime_entries:
                # Prioritize signals by highest prediction probability
                sorted_signals = sorted(yesterday_signals, key=lambda s: s["Prob"], reverse=True)

                for sig in sorted_signals:
                    ticker = sig["Ticker"]

                    if len(active_positions) >= self.max_positions:
                        break
                    if ticker in active_positions:
                        continue

                    price_bar = raw_lookup.get((current_date, ticker))
                    if price_bar is None or float(price_bar.Open) <= 0:
                        continue

                    open_p = float(price_bar.Open)

                    # Mark-to-market estimate for allocation calculation
                    cur_positions_val = sum(
                        p["shares"] * float(raw_lookup.get((current_date, tk), p).Close
                        if hasattr(raw_lookup.get((current_date, tk)), "Close")
                        else p["entry_price"])
                        for tk, p in active_positions.items()
                    )
                    total_portfolio_est = cash + cur_positions_val

                    target_trade_capital = self.position_alloc * total_portfolio_est
                    alloc = min(target_trade_capital, cash)

                    shares = int(alloc / (open_p * (1.0 + self.friction)))
                    if shares <= 0:
                        continue

                    total_cost = shares * open_p * (1.0 + self.friction)
                    if total_cost > cash:
                        continue

                    cash -= total_cost
                    active_positions[ticker] = {
                        "entry_date": current_date,
                        "entry_price": open_p,
                        "shares": shares,
                        "days_held": 0,
                        "max_gain": 0.0,
                    }

            # ------------------------------------------------------------------
            # Step C: End of Day - Mark-to-Market Valuation & Today's Signal Store
            # ------------------------------------------------------------------
            eod_positions_value = 0.0
            for ticker, pos in active_positions.items():
                price_bar = raw_lookup.get((current_date, ticker))
                close_p = float(price_bar.Close) if price_bar is not None else float(pos["entry_price"])
                eod_positions_value += pos["shares"] * close_p

            daily_total_value = cash + eod_positions_value
            bench_val = nifty_lookup.get(current_date, np.nan)

            equity_curve.append({
                "Date": current_date.strftime("%Y-%m-%d"),
                "Strategy_Value": round(daily_total_value, 2),
                "Benchmark_Value": round(bench_val, 2) if not np.isnan(bench_val) else np.nan,
            })

            # Store today's generated signals at Close for tomorrow's execution
            day_group = signals_by_date.get(current_date)
            if day_group is not None:
                day_mask = day_group["Buy_Signal"]
                if "rsi_14" in day_group.columns:
                    day_mask = day_mask & (day_group["rsi_14"] < 0.35)
                if "price_to_sma50" in day_group.columns:
                    day_mask = day_mask & (day_group["price_to_sma50"] > -0.06)
                day_signals = day_group[day_mask]
                yesterday_signals = day_signals.to_dict(orient="records")
            else:
                yesterday_signals = []

        # ----------------------------------------------------------------------
        # Post-Simulation Accounting
        # ----------------------------------------------------------------------
        equity_df = pd.DataFrame(equity_curve)
        if "Benchmark_Value" in equity_df.columns:
            equity_df["Benchmark_Value"] = equity_df["Benchmark_Value"].ffill().bfill()

        trade_log_df = pd.DataFrame(trade_log)
        if trade_log_df.empty:
            trade_log_df = pd.DataFrame(columns=[
                "Ticker", "Entry_Date", "Entry_Price", "Exit_Date", "Exit_Price", "Return_Pct", "Exit_Reason"
            ])

        # Compute Metrics
        strategy_metrics = calculate_performance_metrics(equity_df["Strategy_Value"])
        benchmark_metrics = calculate_performance_metrics(equity_df["Benchmark_Value"])

        # Trade Log Statistics
        total_trades = len(trade_log_df)
        if total_trades > 0:
            win_trades = trade_log_df[trade_log_df["Return_Pct"] > 0]
            loss_trades = trade_log_df[trade_log_df["Return_Pct"] < 0]
            win_rate = (len(win_trades) / total_trades) * 100.0
            gross_win = win_trades["Return_Pct"].sum()
            gross_loss = abs(loss_trades["Return_Pct"].sum())
            profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else (np.inf if gross_win > 0 else 0.0)
        else:
            win_rate = 0.0
            profit_factor = 0.0

        summary = {
            "Strategy": strategy_metrics,
            "Benchmark": benchmark_metrics,
            "Trade_Stats": {
                "Total Trades": total_trades,
                "Win Rate (%)": float(round(win_rate, 2)),
                "Profit Factor": float(round(profit_factor, 2)),
            },
        }

        return equity_df, trade_log_df, summary


# ==============================================================================
# Pipeline Orchestrator & CLI Runner
# ==============================================================================

class BacktestPipeline:
    """
    Coordinates data retrieval, simulation execution, and report generation.
    """

    def __init__(
        self,
        raw_dir: Path = DEFAULT_RAW_DIR,
        processed_dir: Path = DEFAULT_PROCESSED_DIR,
        models_dir: Path = DEFAULT_MODELS_DIR,
        reports_dir: Path = DEFAULT_REPORTS_DIR,
        threshold: float = DEFAULT_PROB_THRESHOLD,
        model_name: Optional[str] = None,
        initial_capital: float = DEFAULT_CAPITAL,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.threshold = threshold
        self.model_name = model_name
        self.initial_capital = initial_capital

    def run(self) -> Dict[str, Any]:
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load Data
        X_test, y_test, raw_nse, indices = load_simulation_data(self.processed_dir, self.raw_dir)

        # 2. Load Model
        model, resolved_name = load_best_model(self.models_dir, self.model_name)
        logger.info("Using model architecture: %s", resolved_name)

        # 3. Simulate
        simulator = TradingSimulator(
            initial_capital=self.initial_capital,
            prob_threshold=self.threshold,
        )
        equity_df, trade_log_df, summary = simulator.run_simulation(
            X_test=X_test,
            raw_prices=raw_nse,
            indices_df=indices,
            model=model,
        )

        # 4. Save Artifacts
        equity_csv_path = self.reports_dir / "equity_curve.csv"
        trade_log_csv_path = self.reports_dir / "trade_log.csv"

        equity_df.to_csv(equity_csv_path, index=False)
        trade_log_df.to_csv(trade_log_csv_path, index=False)

        logger.info("Saved equity curve to %s (%d records)", equity_csv_path, len(equity_df))
        logger.info("Saved trade log to %s (%d executed trades)", trade_log_csv_path, len(trade_log_df))

        # 5. Display Executive Summary Table
        self._print_executive_summary(summary, resolved_name)

        return summary

    def _print_executive_summary(self, summary: Dict[str, Any], model_name: str) -> None:
        strat = summary["Strategy"]
        bench = summary["Benchmark"]
        stats = summary["Trade_Stats"]

        header = f"\n{'='*70}\n NIFTYALPHA REALISTIC BACKTEST PERFORMANCE SUMMARY [{model_name.upper()}]\n{'='*70}"
        table = f"""
{header}
  Metric                       Strategy (NiftyAlpha)     Benchmark (Nifty 50)
  ----------------------------------------------------------------------------
  Total Return (%)             {strat['Total Return (%)']:>18.2f}%    {bench['Total Return (%)']:>18.2f}%
  CAGR (%)                     {strat['CAGR (%)']:>18.2f}%    {bench['CAGR (%)']:>18.2f}%
  Max Drawdown (%)             {strat['Max Drawdown (%)']:>18.2f}%    {bench['Max Drawdown (%)']:>18.2f}%
  Sharpe Ratio (Rf=7%)         {strat['Annualized Sharpe']:>18.2f}     {bench['Annualized Sharpe']:>18.2f}
  ----------------------------------------------------------------------------
  Total Trades Executed:       {stats['Total Trades']:>10}
  Win Rate:                    {stats['Win Rate (%)']:>10.2f}%
  Profit Factor:               {stats['Profit Factor']:>10.2f}
{'='*70}\n"""
        print(table)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NiftyAlpha Realistic Trading Simulation Engine (Phase 5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="Directory containing raw market datasets.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR),
        help="Directory containing processed test parquets.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help="Directory containing serialized model artifacts.",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory where equity curves and trade logs will be stored.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_PROB_THRESHOLD,
        help="Confidence threshold for buy signal generation.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional specific model to test (e.g., xgboost, random_forest).",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=DEFAULT_CAPITAL,
        help="Starting simulation capital in INR.",
    )

    args = parser.parse_args()

    pipeline = BacktestPipeline(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        models_dir=Path(args.models_dir),
        reports_dir=Path(args.reports_dir),
        threshold=args.threshold,
        model_name=args.model_name,
        initial_capital=args.initial_capital,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
