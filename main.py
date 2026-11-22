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
    """Executes Phase 2: Data Ingestion."""
    logger.info(">>> [Phase 2/5] Initiating Multi-Source Data Ingestion...")
    from src.fetch_data import DataIngestionPipeline

    pipeline = DataIngestionPipeline(
        raw_data_dir=RAW_DIR,
        start_date="2014-01-01",
    )
    pipeline.run_all(force=force)
    logger.info(">>> [Phase 2/5] Data Ingestion Completed.")


def run_features() -> None:
    """Executes Phase 3: Feature Engineering & Preprocessing."""
    logger.info(">>> [Phase 3/5] Initiating Feature Engineering & Preprocessing...")
    from src.features import FeatureEngineeringPipeline

    pipeline = FeatureEngineeringPipeline(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        models_dir=MODELS_DIR,
    )
    pipeline.run()
    logger.info(">>> [Phase 3/5] Feature Engineering Completed.")


def run_train() -> None:
    """Executes Phase 4: Machine Learning & Modeling."""
    logger.info(">>> [Phase 4/5] Initiating ML Modeling & Precision Thresholding...")
    from src.models import ModelTrainingPipeline

    pipeline = ModelTrainingPipeline(
        processed_dir=PROCESSED_DIR,
        models_dir=MODELS_DIR,
        reports_dir=REPORTS_DIR,
    )
    pipeline.train_and_evaluate()
    logger.info(">>> [Phase 4/5] Model Training Completed.")


def run_simulate(threshold: float = 0.52) -> None:
    """Executes Phase 5: Realistic Trading Simulation."""
    logger.info(">>> [Phase 5/5] Initiating Day-by-Day Trading Simulation...")
    from src.backtest import BacktestPipeline

    pipeline = BacktestPipeline(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        models_dir=MODELS_DIR,
        reports_dir=REPORTS_DIR,
        threshold=threshold,
    )
    pipeline.run()
    logger.info(">>> [Phase 5/5] Trading Simulation Completed.")


def run_all(force: bool = False, threshold: float = 0.52) -> None:
    """Executes all pipeline phases sequentially."""
    start_time = time.time()
    banner = "=" * 70
    logger.info("\n%s\n  NIFTYALPHA: END-TO-END QUANTITATIVE PIPELINE EXECUTION\n%s", banner, banner)

    run_fetch(force=force)
    run_features()
    run_train()
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
    subparsers.add_parser("train", help="Train and serialize ML models with precision thresholding")

    # Subcommand: simulate
    p_sim = subparsers.add_parser("simulate", help="Run day-by-day backtest simulation on test set")
    p_sim.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold for buy signals")

    # Subcommand: run_all
    p_all = subparsers.add_parser("run_all", help="Execute complete pipeline end-to-end")
    p_all.add_argument("--force", action="store_true", default=False, help="Force fresh download during fetch stage")
    p_all.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold for buy signals")

    args = parser.parse_args()

    if args.command == "fetch":
        run_fetch(force=args.force)
    elif args.command == "features":
        run_features()
    elif args.command == "train":
        run_train()
    elif args.command == "simulate":
        run_simulate(threshold=args.threshold)
    elif args.command == "run_all":
        run_all(force=args.force, threshold=args.threshold)


if __name__ == "__main__":
    main()
