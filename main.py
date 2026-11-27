"""
NiftyAlpha - Quantitative Trading System CLI Orchestrator
=========================================================
Central routing and pipeline orchestration entrypoint for NiftyAlpha.

Subcommands:
- fetch:    Ingest multi-source market and economic data universes.
- features: Extract technical indicators, perform temporal splitting and scaling.
- train:    Train and evaluate machine learning models with precision thresholding.
- simulate: Run out-of-sample backtest simulation with friction and risk controls.
- run_all:  Execute all pipeline phases sequentially end-to-end.
"""

import argparse
import logging
from pathlib import Path
import sys
import time

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NiftyAlpha.Orchestrator")

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def run_fetch(force: bool = False) -> None:
    """Executes Data Ingestion pipeline."""
    logger.info(">>> [Data Ingestion] Initiating Multi-Source Ingestion...")
    from src.fetch_data import DataIngestionPipeline

    pipeline = DataIngestionPipeline(
        raw_data_dir=RAW_DIR,
        start_date="2014-01-01",
    )
    pipeline.run_all(force=force)
    logger.info(">>> [Data Ingestion] Completed.")


def run_features() -> None:
    """Executes Feature Engineering & Preprocessing."""
    logger.info(">>> [Feature Engineering] Initiating Feature Engineering & Preprocessing...")
    from src.features import FeatureEngineeringPipeline

    pipeline = FeatureEngineeringPipeline(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        models_dir=MODELS_DIR,
    )
    pipeline.run()
    logger.info(">>> [Feature Engineering] Completed.")


def run_train(
    tune: bool = False,
    tune_model: str = "xgboost",
    n_splits: int = 5,
    n_iter: int = 8,
) -> None:
    """Executes Machine Learning & Modeling."""
    logger.info(">>> [Model Training] Initiating ML Modeling & Precision Thresholding (Tune=%s)...", tune)
    from src.models import ModelTrainingPipeline

    pipeline = ModelTrainingPipeline(
        processed_dir=PROCESSED_DIR,
        models_dir=MODELS_DIR,
        reports_dir=REPORTS_DIR,
        tune=tune,
        tune_model=tune_model,
        n_splits=n_splits,
        n_iter=n_iter,
    )
    pipeline.train_and_evaluate()
    logger.info(">>> [Model Training] Completed.")


def run_simulate(threshold: float = 0.525) -> None:
    """Executes Realistic Trading Simulation."""
    logger.info(">>> [Simulation] Initiating Day-by-Day Trading Simulation...")
    from src.backtest import BacktestPipeline

    pipeline = BacktestPipeline(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        models_dir=MODELS_DIR,
        reports_dir=REPORTS_DIR,
        threshold=threshold,
    )
    pipeline.run()
    logger.info(">>> [Simulation] Completed.")


def run_predict(
    date: str = None,
    threshold: float = 0.50,
    model: str = "xgboost",
) -> None:
    """Executes Automated Daily Predictor & Order Generator."""
    logger.info(">>> [Daily Prediction] Initiating Daily Order Generation (Model=%s, Thresh=%.2f)...", model, threshold)
    from scripts.daily_predict import DailyPredictor

    predictor = DailyPredictor(
        processed_dir=PROCESSED_DIR,
        raw_dir=RAW_DIR,
        models_dir=MODELS_DIR,
        reports_dir=REPORTS_DIR,
        model_name=model,
        threshold=threshold,
    )
    predictor.run(target_date=date)
    logger.info(">>> [Daily Prediction] Completed.")


def run_all(
    force: bool = False,
    threshold: float = 0.525,
    tune: bool = False,
    tune_model: str = "xgboost",
) -> None:
    """Executes all pipeline phases sequentially."""
    start_time = time.time()
    banner = "=" * 70
    logger.info("\n%s\n  NIFTYALPHA: END-TO-END QUANTITATIVE PIPELINE EXECUTION\n%s", banner, banner)

    run_fetch(force=force)
    run_features()
    run_train(tune=tune, tune_model=tune_model)
    run_simulate(threshold=threshold)

    elapsed = time.time() - start_time
    logger.info("\n%s\n  PIPELINE EXECUTION COMPLETED IN %.2f SECONDS\n%s", banner, elapsed, banner)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NiftyAlpha Quantitative Trading System Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Subcommands",
        description="Select pipeline phase to execute",
        required=True,
    )

    # Subcommand: fetch
    p_fetch = subparsers.add_parser("fetch", help="Ingest multi-source market and economic data")
    p_fetch.add_argument("--force", action="store_true", default=False, help="Force fresh download ignoring cache")

    # Subcommand: features
    subparsers.add_parser("features", help="Extract technical features, temporal split, and scale data")

    # Subcommand: train
    p_train = subparsers.add_parser("train", help="Train and serialize ML models with precision thresholding")
    p_train.add_argument("--tune", action="store_true", default=False, help="Run automated hyperparameter tuning with TimeSeriesSplit")
    p_train.add_argument("--tune-model", type=str, default="xgboost", help="Model architecture to tune (e.g. xgboost, random_forest)")
    p_train.add_argument("--n-splits", type=int, default=5, help="Number of chronological splits for TimeSeriesSplit")
    p_train.add_argument("--n-iter", type=int, default=8, help="Number of parameter search iterations for RandomizedSearchCV")

    # Subcommand: simulate
    p_sim = subparsers.add_parser("simulate", help="Run day-by-day backtest simulation on test set")
    p_sim.add_argument("--threshold", type=float, default=0.525, help="Confidence threshold for buy signals")

    # Subcommand: predict (Automated Cron / Daily Order Generation)
    p_pred = subparsers.add_parser("predict", help="Generate actionable daily trade orders for tomorrow")
    p_pred.add_argument("--date", type=str, default=None, help="Target market date ('YYYY-MM-DD')")
    p_pred.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold for buy orders")
    p_pred.add_argument("--model", type=str, default="xgboost", help="Model architecture to use")

    # Subcommand: run_all
    p_all = subparsers.add_parser("run_all", help="Execute complete pipeline end-to-end")
    p_all.add_argument("--force", action="store_true", default=False, help="Force fresh download during fetch stage")
    p_all.add_argument("--threshold", type=float, default=0.525, help="Confidence threshold for buy signals")
    p_all.add_argument("--tune", action="store_true", default=False, help="Run automated hyperparameter tuning during train stage")
    p_all.add_argument("--tune-model", type=str, default="xgboost", help="Model architecture to tune")

    args = parser.parse_args()

    if args.command == "fetch":
        run_fetch(force=args.force)
    elif args.command == "features":
        run_features()
    elif args.command == "train":
        run_train(
            tune=args.tune,
            tune_model=args.tune_model,
            n_splits=args.n_splits,
            n_iter=args.n_iter,
        )
    elif args.command == "simulate":
        run_simulate(threshold=args.threshold)
    elif args.command == "predict":
        run_predict(
            date=args.date,
            threshold=args.threshold,
            model=args.model,
        )
    elif args.command == "run_all":
        run_all(
            force=args.force,
            threshold=args.threshold,
            tune=args.tune,
            tune_model=args.tune_model,
        )


if __name__ == "__main__":
    main()
