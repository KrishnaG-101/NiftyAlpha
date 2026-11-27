"""
NiftyAlpha - Quantitative Machine Learning & Algorithmic Trading Framework
==========================================================================
Institutional quantitative equity trading and risk management platform for
the National Stock Exchange of India (NSE).
"""

from src.fetch_data import DataIngestionPipeline
from src.features import FeatureEngineeringPipeline
from src.models import ModelTrainingPipeline
from src.backtest import BacktestPipeline

__all__ = [
    "DataIngestionPipeline",
    "FeatureEngineeringPipeline",
    "ModelTrainingPipeline",
    "BacktestPipeline",
]

__version__ = "1.0.0"
