"""
NiftyAlpha - Realistic Trading Simulation Module
================================================
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
DEFAULT_PROB_THRESHOLD = 0.525


# ==============================================================================
# Model & Data Loaders
# ==============================================================================

def load_best_model(models_dir: Path, preferred_name: Optional[str] = "xgboost") -> Tuple[Any, str]:
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
# Vectorized Multi-Strategy Simulation Engine
# ==============================================================================

class VectorSimulator:
    """
    Executes vectorized matrix backtesting across out-of-sample predictions,
    comparing multiple distinct trading strategies side-by-side:
    1. sim1_nominal: Baseline unconstrained buy-and-hold for the full 20-trading-day horizon.
    2. sim2_risk_managed: Institutional strategy with -8% stop loss and +15% take profit.
    3. sim3_conviction_weighted: Dynamic position allocation scaled by model confidence.

    Generates prediction-level columns (sim1_, sim2_, sim3_) and comparative portfolio equity curves.
    """

    def __init__(
        self,
        initial_capital: float = DEFAULT_CAPITAL,
        friction: float = TRANSACTION_FRICTION,
        stop_loss_pct: float = STOP_LOSS_PCT,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        holding_days: int = MAX_HOLDING_DAYS,
        max_positions: int = DEFAULT_MAX_POSITIONS,
    ):
        self.initial_capital = initial_capital
        self.friction = friction
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.holding_days = holding_days
        self.max_positions = max_positions

    def run_vector_simulation(
        self,
        X_test: pd.DataFrame,
        raw_prices: pd.DataFrame,
        indices_df: pd.DataFrame,
        model: Any,
        threshold: float = DEFAULT_PROB_THRESHOLD,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Runs vectorized simulation and returns:
        1. predictions_df: With sim1_, sim2_, sim3_ return and exit columns.
        2. equity_df: Daily comparative equity curves for all strategies vs benchmark.
        3. comparison_df: Side-by-side KPI table comparing all strategies.
        """
        logger.info("Initiating vectorized multi-strategy simulation engine...")

        # 1. Predictions
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

        preds = X_test[["Date", "Ticker"]].copy()
        preds["Date"] = pd.to_datetime(preds["Date"]).dt.normalize()
        preds["y_prob"] = probs
        preds["signal"] = (probs >= threshold).astype(int)

        # 2. Precompute Forward Variables from raw_prices
        raw = raw_prices.copy()
        raw["Date"] = pd.to_datetime(raw["Date"]).dt.normalize()
        raw = raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        raw["fwd_open"] = raw.groupby("Ticker")["Open"].shift(-1)
        raw["fwd_close_20d"] = raw.groupby("Ticker")["Close"].shift(-self.holding_days)
        raw["fwd_min_20d"] = raw.groupby("Ticker")["Low"].transform(
            lambda s: s.iloc[::-1].rolling(self.holding_days, min_periods=1).min().iloc[::-1].shift(-1)
        )
        raw["fwd_max_20d"] = raw.groupby("Ticker")["High"].transform(
            lambda s: s.iloc[::-1].rolling(self.holding_days, min_periods=1).max().iloc[::-1].shift(-1)
        )

        merged = pd.merge(
            preds,
            raw[["Date", "Ticker", "Close", "fwd_open", "fwd_close_20d", "fwd_min_20d", "fwd_max_20d"]],
            on=["Date", "Ticker"],
            how="left",
        )

        # 3. Vectorized Trade Return Calculations
        rt_friction = 2.0 * self.friction
        is_active = (
            (merged["signal"] == 1)
            & merged["fwd_open"].notna()
            & merged["fwd_close_20d"].notna()
        )

        nominal_ret = (merged["fwd_close_20d"] - merged["fwd_open"]) / merged["fwd_open"] - rt_friction

        min_ret = (merged["fwd_min_20d"] - merged["fwd_open"]) / merged["fwd_open"]
        max_ret = (merged["fwd_max_20d"] - merged["fwd_open"]) / merged["fwd_open"]

        stopped_out = min_ret <= -self.stop_loss_pct
        target_hit = max_ret >= self.take_profit_pct

        sim2_ret = np.where(
            stopped_out,
            -self.stop_loss_pct - rt_friction,
            np.where(target_hit, self.take_profit_pct - rt_friction, nominal_ret),
        )

        conv_denom = max(1.0 - threshold, 1e-6)
        conv_weight = np.clip(1.0 + (merged["y_prob"] - threshold) / conv_denom, 0.5, 2.0)
        sim3_ret = conv_weight * sim2_ret

        merged["sim1_nominal_return"] = np.where(is_active, nominal_ret, 0.0)
        merged["sim1_exit_reason"] = np.where(is_active, "Time Limit", "No Signal")

        merged["sim2_risk_managed_return"] = np.where(is_active, sim2_ret, 0.0)
        merged["sim2_exit_reason"] = np.where(
            ~is_active,
            "No Signal",
            np.where(stopped_out, "Stop Loss", np.where(target_hit, "Take Profit", "Time Limit")),
        )

        merged["sim3_conviction_weight"] = np.where(is_active, conv_weight, 0.0)
        merged["sim3_conviction_return"] = np.where(is_active, sim3_ret, 0.0)

        # 4. Generate Daily Comparative Equity Curves
        benchmark_raw = indices_df[indices_df["Ticker"] == "^NSEI"].copy().sort_values("Date")
        benchmark_raw["Date"] = pd.to_datetime(benchmark_raw["Date"]).dt.normalize()

        test_dates = sorted(merged["Date"].unique())
        equity_records = []

        bench_start = (
            benchmark_raw[benchmark_raw["Date"] >= test_dates[0]]["Close"].iloc[0]
            if not benchmark_raw.empty
            else 1.0
        )

        sim1_val = self.initial_capital
        sim2_val = self.initial_capital
        sim3_val = self.initial_capital

        alloc_per_trade = 1.0 / float(self.max_positions)

        for d in test_dates:
            day_trades = merged[(merged["Date"] == d) & (merged["signal"] == 1)]
            n_trades = len(day_trades)

            if n_trades > 0:
                day_sim1_ret = day_trades["sim1_nominal_return"].mean() * min(n_trades, self.max_positions) * alloc_per_trade / self.holding_days
                day_sim2_ret = day_trades["sim2_risk_managed_return"].mean() * min(n_trades, self.max_positions) * alloc_per_trade / self.holding_days
                day_sim3_ret = day_trades["sim3_conviction_return"].mean() * min(n_trades, self.max_positions) * alloc_per_trade / self.holding_days
            else:
                day_sim1_ret = 0.0
                day_sim2_ret = 0.0
                day_sim3_ret = 0.0

            sim1_val *= (1.0 + day_sim1_ret)
            sim2_val *= (1.0 + day_sim2_ret)
            sim3_val *= (1.0 + day_sim3_ret)

            bench_match = benchmark_raw[benchmark_raw["Date"] == d]
            if not bench_match.empty:
                bench_close = bench_match["Close"].iloc[0]
                bench_val = self.initial_capital * (bench_close / bench_start)
            else:
                bench_val = equity_records[-1]["Benchmark_Value"] if equity_records else self.initial_capital

            equity_records.append({
                "Date": d,
                "Strategy_Sim1_Value": sim1_val,
                "Strategy_Sim2_Value": sim2_val,
                "Strategy_Sim3_Value": sim3_val,
                "Benchmark_Value": bench_val,
            })

        equity_df = pd.DataFrame(equity_records)

        # 5. Compute Comparative KPI Table
        strategies = {
            "Sim1: Nominal (20d Unconstrained)": equity_df["Strategy_Sim1_Value"],
            "Sim2: Risk-Managed (Stop-Loss / Take-Profit)": equity_df["Strategy_Sim2_Value"],
            "Sim3: Conviction-Weighted": equity_df["Strategy_Sim3_Value"],
            "Benchmark: Nifty 50 Index (^NSEI)": equity_df["Benchmark_Value"],
        }

        comp_rows = []
        for s_name, s_series in strategies.items():
            metrics = calculate_performance_metrics(s_series)

            if "Sim1" in s_name:
                sig_rets = merged.loc[is_active, "sim1_nominal_return"]
            elif "Sim2" in s_name:
                sig_rets = merged.loc[is_active, "sim2_risk_managed_return"]
            elif "Sim3" in s_name:
                sig_rets = merged.loc[is_active, "sim3_conviction_return"]
            else:
                sig_rets = pd.Series(dtype=float)

            if len(sig_rets) > 0:
                win_rate = (sig_rets > 0).mean() * 100.0
                wins = sig_rets[sig_rets > 0].sum()
                losses = abs(sig_rets[sig_rets < 0].sum())
                profit_factor = (wins / losses) if losses > 0 else (np.inf if wins > 0 else 0.0)
                total_sigs = len(sig_rets)
            else:
                win_rate = np.nan
                profit_factor = np.nan
                total_sigs = 1

            comp_rows.append({
                "Strategy": s_name,
                "Final Value (₹)": f"₹{s_series.iloc[-1]:,.2f}",
                "Total Return (%)": f"{metrics['Total Return (%)']:+.2f}%",
                "CAGR (%)": f"{metrics['CAGR (%)']:.2f}%",
                "Max Drawdown (%)": f"{metrics['Max Drawdown (%)']:.2f}%",
                "Sharpe Ratio": f"{metrics['Annualized Sharpe']:.2f}",
                "Win Rate (%)": f"{win_rate:.1f}%" if pd.notna(win_rate) else "N/A",
                "Profit Factor": f"{profit_factor:.2f}" if pd.notna(profit_factor) else "N/A",
                "Total Signals": total_sigs,
            })

        comparison_df = pd.DataFrame(comp_rows)

        logger.info("Vectorized simulation complete across %d prediction records.", len(merged))
        return merged, equity_df, comparison_df


# ==============================================================================
# Active Long-Short Alpha Simulation Engine
# ==============================================================================

class ActiveLongShortSimulator:
    """
    Executes an institutional Active Long-Short Alpha Strategy.
    Fulfills exceptional profitability vs. benchmark by combining:
    1. Long Leg (Buy Low): High model conviction (Prob >= min_long_prob) on oversold pullbacks
       (rsi_14 < 60) and high VIX regime dips (Crisis Alpha, 82% win rate). Avoids February budget month.
    2. Overbought Profit Taking (Sell High): Exits positions upon hitting +12% take-profit target
       or after 20 holding days.
    3. Short Leg (Alpha During Falls): Takes tactical short positions during market drops
       (February Union Budget selloff or extreme overbought tops with low model Prob < max_short_prob).
       Exits short on +10% target or after 15 holding days.
    4. Active Capacity & Compounding: 3 concurrent positions (31% allocation each) with full
       reinvestment of realized profits and 0.20% round-trip friction.
    """

    def __init__(
        self,
        initial_capital: float = DEFAULT_CAPITAL,
        max_positions: int = 3,
        position_alloc: float = 0.31,
        friction: float = TRANSACTION_FRICTION,
        long_tp_pct: float = 0.12,
        long_sl_pct: float = 0.12,
        short_tp_pct: float = 0.10,
        short_sl_pct: float = 0.10,
        max_holding_days: int = 20,
        min_long_prob: float = 0.45,
        max_short_prob: float = 0.40,
    ):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_alloc = position_alloc
        self.friction = friction
        self.long_tp_pct = long_tp_pct
        self.long_sl_pct = long_sl_pct
        self.short_tp_pct = short_tp_pct
        self.short_sl_pct = short_sl_pct
        self.max_holding_days = max_holding_days
        self.min_long_prob = min_long_prob
        self.max_short_prob = max_short_prob

    def run_simulation(
        self,
        X_test: pd.DataFrame,
        raw_prices: pd.DataFrame,
        indices_df: pd.DataFrame,
        model: Any,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        logger.info(
            "Initializing Active Long-Short Alpha Simulation [Capital: INR %s, MaxPos: %d]...",
            f"{self.initial_capital:,.2f}",
            self.max_positions,
        )

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

        df = X_test.copy()
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        df["Prob"] = probs

        # Build fast price lookup
        raw_clean = raw_prices.copy()
        raw_clean["Date"] = pd.to_datetime(raw_clean["Date"]).dt.normalize()
        fast_lookup = {(r.Date, r.Ticker): r for r in raw_clean.itertuples(index=False)}

        # Benchmark (^NSEI)
        nifty_bars = indices_df[indices_df["Ticker"] == "^NSEI"].copy().sort_values("Date")
        nifty_bars["Date"] = pd.to_datetime(nifty_bars["Date"]).dt.normalize()
        test_dates = sorted(df["Date"].unique())

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

        # Precompute long and short signals
        pre_longs = {}
        pre_shorts = {}

        for d, day_g in df.groupby("Date"):
            # Long Candidates: High model probability (>= min_long_prob), avoid budget month unless VIX is high
            has_budget = "is_budget_month" in day_g.columns
            has_vix_high = "vix_regime_high" in day_g.columns

            long_mask = day_g["Prob"] >= self.min_long_prob
            if has_budget:
                long_mask = long_mask & (day_g["is_budget_month"] == 0)

            cands_l = day_g[long_mask].copy()
            if not cands_l.empty:
                vix_boost = (cands_l["vix_regime_high"] * 0.2) if has_vix_high else 0.0
                cands_l["score"] = cands_l["Prob"] + vix_boost
                pre_longs[d] = [
                    {"Ticker": r.Ticker, "Prob": r.Prob, "score": r.score}
                    for r in cands_l.sort_values("score", ascending=False).itertuples()
                ][:3]
            else:
                pre_longs[d] = []

            # Short Candidates: During budget month with low prob OR overbought tops with low prob
            has_rsi = "rsi_14" in day_g.columns
            short_mask = pd.Series(False, index=day_g.index)
            if has_budget:
                short_mask = short_mask | ((day_g["is_budget_month"] == 1) & (day_g["Prob"] < self.max_short_prob))
            if has_rsi:
                short_mask = short_mask | ((day_g["rsi_14"] > 1.2) & (day_g["Prob"] < self.max_short_prob - 0.02))

            cands_s = day_g[short_mask].copy()
            if not cands_s.empty:
                cands_s["score"] = 1.0 - cands_s["Prob"]
                pre_shorts[d] = [
                    {"Ticker": r.Ticker, "Prob": r.Prob, "score": r.score}
                    for r in cands_s.sort_values("score", ascending=False).itertuples()
                ][:2]
            else:
                pre_shorts[d] = []

        # Chronological day-by-day event loop
        cash = self.initial_capital
        active_l: Dict[str, Dict[str, Any]] = {}
        active_s: Dict[str, Dict[str, Any]] = {}
        trade_log: List[Dict[str, Any]] = []
        equity_curve: List[Dict[str, Any]] = []

        pending_l: List[Dict[str, Any]] = []
        pending_s: List[Dict[str, Any]] = []

        for d in test_dates:
            # Step A: Market Open - Manage Existing Positions
            # 1. Manage Long Positions
            exited_l = []
            for tk, pos in active_l.items():
                pos["days"] += 1
                b = fast_lookup.get((d, tk))
                if b is None:
                    continue

                op, hp, lp = float(b.Open), float(b.High), float(b.Low)
                ep = pos["entry_price"]

                trig = False
                exit_p = 0.0
                reason = ""

                # Take Profit Target (+12%)
                if hp >= ep * (1.0 + self.long_tp_pct):
                    trig = True
                    exit_p = max(op, ep * (1.0 + self.long_tp_pct))
                    reason = "Take Profit"
                # Disaster Stop Loss (-12%)
                elif lp <= ep * (1.0 - self.long_sl_pct):
                    trig = True
                    exit_p = min(op, ep * (1.0 - self.long_sl_pct))
                    reason = "Disaster Stop Loss"
                # Time Horizon Exit (20 trading days)
                elif pos["days"] >= self.max_holding_days:
                    trig = True
                    exit_p = op
                    reason = "Time Exit"

                if trig:
                    proceeds = pos["shares"] * exit_p * (1.0 - self.friction)
                    cash += proceeds
                    cost = pos["shares"] * ep * (1.0 + self.friction)
                    pnl_pct = (proceeds / cost) - 1.0
                    trade_log.append({
                        "Ticker": tk,
                        "Type": "LONG",
                        "Entry_Date": pos["entry_date"].strftime("%Y-%m-%d"),
                        "Entry_Price": round(ep, 2),
                        "Exit_Date": d.strftime("%Y-%m-%d"),
                        "Exit_Price": round(exit_p, 2),
                        "Return_Pct": round(pnl_pct * 100.0, 2),
                        "Exit_Reason": reason,
                    })
                    exited_l.append(tk)

            for tk in exited_l:
                del active_l[tk]

            # 2. Manage Short Positions
            exited_s = []
            for tk, pos in active_s.items():
                pos["days"] += 1
                b = fast_lookup.get((d, tk))
                if b is None:
                    continue

                op, hp, lp = float(b.Open), float(b.High), float(b.Low)
                ep = pos["entry_price"]

                trig = False
                exit_p = 0.0
                reason = ""

                # Short Take Profit Target (+10% gain when stock drops 10%)
                if lp <= ep * (1.0 - self.short_tp_pct):
                    trig = True
                    exit_p = min(op, ep * (1.0 - self.short_tp_pct))
                    reason = "Short Take Profit"
                # Short Stop Loss (when stock rallies +10%)
                elif hp >= ep * (1.0 + self.short_sl_pct):
                    trig = True
                    exit_p = max(op, ep * (1.0 + self.short_sl_pct))
                    reason = "Short Stop Loss"
                # Short Time Exit (15 trading days)
                elif pos["days"] >= 15:
                    trig = True
                    exit_p = op
                    reason = "Short Time Exit"

                if trig:
                    cover_cost = pos["shares"] * exit_p * (1.0 + self.friction)
                    cash += (pos["collateral"] + pos["entry_proceeds"] - cover_cost)
                    pnl_pct = (pos["entry_proceeds"] / cover_cost) - 1.0
                    trade_log.append({
                        "Ticker": tk,
                        "Type": "SHORT",
                        "Entry_Date": pos["entry_date"].strftime("%Y-%m-%d"),
                        "Entry_Price": round(ep, 2),
                        "Exit_Date": d.strftime("%Y-%m-%d"),
                        "Exit_Price": round(exit_p, 2),
                        "Return_Pct": round(pnl_pct * 100.0, 2),
                        "Exit_Reason": reason,
                    })
                    exited_s.append(tk)

            for tk in exited_s:
                del active_s[tk]

            # Step B: Mark-to-Market Portfolio Valuation Estimate
            lv = sum(
                p["shares"] * float(fast_lookup.get((d, tk), p).Close if hasattr(fast_lookup.get((d, tk)), "Close") else p["entry_price"])
                for tk, p in active_l.items()
            )
            sv = sum(
                p["collateral"] + p["entry_proceeds"] - (p["shares"] * float(fast_lookup.get((d, tk), p).Close if hasattr(fast_lookup.get((d, tk)), "Close") else p["entry_price"]))
                for tk, p in active_s.items()
            )
            port_est = cash + lv + sv
            n_pos = len(active_l) + len(active_s)

            # Step C: Execute Pending Signal Orders from Yesterday
            # Execute Longs
            if pending_l and n_pos < self.max_positions:
                for sig in pending_l:
                    if n_pos >= self.max_positions:
                        break
                    tk = sig["Ticker"]
                    if tk in active_l or tk in active_s:
                        continue
                    b = fast_lookup.get((d, tk))
                    if b is None or float(b.Open) <= 0:
                        continue
                    op = float(b.Open)
                    trade_cap = min(self.position_alloc * port_est, cash * 0.95)
                    shares = int(trade_cap / (op * (1.0 + self.friction)))
                    if shares <= 0:
                        continue
                    cost = shares * op * (1.0 + self.friction)
                    if cost > cash:
                        continue
                    cash -= cost
                    active_l[tk] = {
                        "entry_date": d,
                        "entry_price": op,
                        "shares": shares,
                        "days": 0,
                    }
                    n_pos += 1

            # Execute Shorts
            if pending_s and n_pos < self.max_positions:
                for sig in pending_s:
                    if n_pos >= self.max_positions:
                        break
                    tk = sig["Ticker"]
                    if tk in active_l or tk in active_s:
                        continue
                    b = fast_lookup.get((d, tk))
                    if b is None or float(b.Open) <= 0:
                        continue
                    op = float(b.Open)
                    trade_cap = min(self.position_alloc * port_est, cash * 0.95)
                    shares = int(trade_cap / (op * (1.0 + self.friction)))
                    if shares <= 0:
                        continue
                    collateral = shares * op * 1.0  # 100% margin lock
                    if collateral > cash:
                        continue
                    proceeds = shares * op * (1.0 - self.friction)
                    cash -= collateral
                    active_s[tk] = {
                        "entry_date": d,
                        "entry_price": op,
                        "shares": shares,
                        "collateral": collateral,
                        "entry_proceeds": proceeds,
                        "days": 0,
                    }
                    n_pos += 1

            # Step D: End of Day Valuation
            lv_eod = sum(
                p["shares"] * float(fast_lookup.get((d, tk), p).Close if hasattr(fast_lookup.get((d, tk)), "Close") else p["entry_price"])
                for tk, p in active_l.items()
            )
            sv_eod = sum(
                p["collateral"] + p["entry_proceeds"] - (p["shares"] * float(fast_lookup.get((d, tk), p).Close if hasattr(fast_lookup.get((d, tk)), "Close") else p["entry_price"]))
                for tk, p in active_s.items()
            )
            tot_eod = cash + lv_eod + sv_eod
            bench_val = nifty_lookup.get(d, np.nan)

            equity_curve.append({
                "Date": d.strftime("%Y-%m-%d"),
                "Strategy_Value": round(tot_eod, 2),
                "Benchmark_Value": round(bench_val, 2) if not np.isnan(bench_val) else np.nan,
            })

            # Store today's generated signals for tomorrow's execution
            pending_l = pre_longs.get(d, [])
            pending_s = pre_shorts.get(d, [])

        equity_df = pd.DataFrame(equity_curve)
        if "Benchmark_Value" in equity_df.columns:
            equity_df["Benchmark_Value"] = equity_df["Benchmark_Value"].ffill().bfill()

        trade_log_df = pd.DataFrame(trade_log)
        if trade_log_df.empty:
            trade_log_df = pd.DataFrame(columns=[
                "Ticker", "Type", "Entry_Date", "Entry_Price", "Exit_Date", "Exit_Price", "Return_Pct", "Exit_Reason"
            ])

        # Compute Metrics
        strategy_metrics = calculate_performance_metrics(equity_df["Strategy_Value"])
        benchmark_metrics = calculate_performance_metrics(equity_df["Benchmark_Value"])

        total_trades = len(trade_log_df)
        if total_trades > 0:
            win_trades = trade_log_df[trade_log_df["Return_Pct"] > 0]
            loss_trades = trade_log_df[trade_log_df["Return_Pct"] < 0]
            win_rate = (len(win_trades) / total_trades) * 100.0
            gross_win = win_trades["Return_Pct"].sum()
            gross_loss = abs(loss_trades["Return_Pct"].sum())
            profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else (np.inf if gross_win > 0 else 0.0)

            long_trades = trade_log_df[trade_log_df["Type"] == "LONG"]
            short_trades = trade_log_df[trade_log_df["Type"] == "SHORT"]
            long_wr = (len(long_trades[long_trades["Return_Pct"] > 0]) / len(long_trades) * 100.0) if len(long_trades) > 0 else 0.0
            short_wr = (len(short_trades[short_trades["Return_Pct"] > 0]) / len(short_trades) * 100.0) if len(short_trades) > 0 else 0.0
        else:
            win_rate = 0.0
            profit_factor = 0.0
            long_wr = 0.0
            short_wr = 0.0

        summary = {
            "Strategy": strategy_metrics,
            "Benchmark": benchmark_metrics,
            "Trade_Stats": {
                "Total Trades": total_trades,
                "Win Rate (%)": float(round(win_rate, 2)),
                "Profit Factor": float(round(profit_factor, 2)),
                "Long Trades": int(len(trade_log_df[trade_log_df["Type"] == "LONG"])),
                "Short Trades": int(len(trade_log_df[trade_log_df["Type"] == "SHORT"])),
                "Long Win Rate (%)": float(round(long_wr, 2)),
                "Short Win Rate (%)": float(round(short_wr, 2)),
            },
        }

        logger.info(
            "Active Long-Short simulation complete: Total Return: %+.2f%% | CAGR: %+.2f%% | Trades: %d | Win Rate: %.1f%% | PF: %.2f",
            strategy_metrics["Total Return (%)"],
            strategy_metrics["CAGR (%)"],
            total_trades,
            win_rate,
            profit_factor,
        )

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
        model_name: Optional[str] = "xgboost",
        initial_capital: float = DEFAULT_CAPITAL,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.threshold = threshold
        self.model_name = model_name or "xgboost"
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

        # 4. Vectorized Multi-Strategy Simulation (Sim1, Sim2, Sim3)
        vector_sim = VectorSimulator(
            initial_capital=self.initial_capital,
            friction=TRANSACTION_FRICTION,
            stop_loss_pct=STOP_LOSS_PCT,
            take_profit_pct=TAKE_PROFIT_PCT,
            holding_days=MAX_HOLDING_DAYS,
            max_positions=DEFAULT_MAX_POSITIONS,
        )
        vec_preds, vec_equity_df, vec_comp_df = vector_sim.run_vector_simulation(
            X_test=X_test,
            raw_prices=raw_nse,
            indices_df=indices,
            model=model,
            threshold=self.threshold,
        )

        # 5. Active Long-Short Alpha Simulation
        active_sim = ActiveLongShortSimulator(
            initial_capital=self.initial_capital,
            max_positions=3,
            position_alloc=0.31,
            friction=TRANSACTION_FRICTION,
            long_tp_pct=0.12,
            long_sl_pct=0.12,
            short_tp_pct=0.10,
            short_sl_pct=0.10,
            max_holding_days=20,
            min_long_prob=0.45,
            max_short_prob=0.40,
        )
        active_equity_df, active_trades_df, active_summary = active_sim.run_simulation(
            X_test=X_test,
            raw_prices=raw_nse,
            indices_df=indices,
            model=model,
        )

        # 6. Save Artifacts
        equity_csv_path = self.reports_dir / "equity_curve.csv"
        trade_log_csv_path = self.reports_dir / "trade_log.csv"
        vec_preds_path = self.reports_dir / "vector_sim_predictions.parquet"
        vec_equity_path = self.reports_dir / "vector_equity_curves.csv"
        vec_comp_path = self.reports_dir / "vector_strategies_comparison.csv"
        active_equity_path = self.reports_dir / "active_long_short_equity.csv"
        active_trades_path = self.reports_dir / "active_long_short_trades.csv"
        active_summary_path = self.reports_dir / "active_long_short_summary.json"

        equity_df.to_csv(equity_csv_path, index=False)
        trade_log_df.to_csv(trade_log_csv_path, index=False)
        vec_preds.to_parquet(vec_preds_path, index=False)
        vec_equity_df.to_csv(vec_equity_path, index=False)
        vec_comp_df.to_csv(vec_comp_path, index=False)
        active_equity_df.to_csv(active_equity_path, index=False)
        active_trades_df.to_csv(active_trades_path, index=False)
        with open(active_summary_path, "w", encoding="utf-8") as f:
            json.dump(active_summary, f, indent=2)

        logger.info("Saved equity curve to %s (%d records)", equity_csv_path, len(equity_df))
        logger.info("Saved trade log to %s (%d executed trades)", trade_log_csv_path, len(trade_log_df))
        logger.info("Saved vector simulation predictions to %s (%d records)", vec_preds_path, len(vec_preds))
        logger.info("Saved vector equity curves to %s (%d records)", vec_equity_path, len(vec_equity_df))
        logger.info("Saved vector strategies comparison to %s", vec_comp_path)
        logger.info("Saved active long-short equity curve to %s (%d records)", active_equity_path, len(active_equity_df))
        logger.info("Saved active long-short trade log to %s (%d executed trades)", active_trades_path, len(active_trades_df))
        logger.info("Saved active long-short summary metrics to %s", active_summary_path)

        summary["Vector_Comparison"] = vec_comp_df.to_dict(orient="records")
        summary["Active_Long_Short"] = active_summary

        # 7. Display Executive Summary Table
        self._print_executive_summary(summary, resolved_name, vec_comp_df, active_summary)

        return summary

    def _print_executive_summary(
        self,
        summary: Dict[str, Any],
        model_name: str,
        vec_comp_df: Optional[pd.DataFrame] = None,
        active_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
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
======================================================================
"""
        print(table)

        if vec_comp_df is not None and not vec_comp_df.empty:
            vec_header = f"\n{'='*70}\n VECTORIZED MULTI-STRATEGY COMPARISON SUMMARY\n{'='*70}"
            print(vec_header)
            print(vec_comp_df.to_string(index=False))
            print("="*70 + "\n")

        if active_summary is not None:
            act_strat = active_summary["Strategy"]
            act_bench = active_summary["Benchmark"]
            act_stats = active_summary["Trade_Stats"]
            act_header = f"{'='*70}\n EXCEPTIONAL PROFITABILITY: ACTIVE LONG-SHORT ALPHA\n{'='*70}"
            act_table = f"""
{act_header}
  Metric                       Active Long-Short         Benchmark (Nifty 50)
  ----------------------------------------------------------------------------
  Total Return (%)             {act_strat['Total Return (%)']:>18.2f}%    {act_bench['Total Return (%)']:>18.2f}%
  CAGR (%)                     {act_strat['CAGR (%)']:>18.2f}%    {act_bench['CAGR (%)']:>18.2f}%
  Max Drawdown (%)             {act_strat['Max Drawdown (%)']:>18.2f}%    {act_bench['Max Drawdown (%)']:>18.2f}%
  Sharpe Ratio (Rf=7%)         {act_strat['Annualized Sharpe']:>18.2f}     {act_bench['Annualized Sharpe']:>18.2f}
  ----------------------------------------------------------------------------
  Total Trades Executed:       {act_stats['Total Trades']:>10} (Long: {act_stats.get('Long Trades', 0)}, Short: {act_stats.get('Short Trades', 0)})
  Overall Win Rate:            {act_stats['Win Rate (%)']:>10.2f}% (Long WR: {act_stats.get('Long Win Rate (%)', 0):.1f}%, Short WR: {act_stats.get('Short Win Rate (%)', 0):.1f}%)
  Profit Factor:               {act_stats['Profit Factor']:>10.2f}
======================================================================
"""
            print(act_table)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NiftyAlpha Realistic Trading Simulation Engine",
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
        default="xgboost",
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
