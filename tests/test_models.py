"""
Unit tests for src/models.py
============================
Validates model initialization, metadata separation, precision thresholding,
model serialization, and metrics generation.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models import (
    get_model_instances,
    evaluate_model_at_thresholds,
    get_shap_explainer,
    ModelTrainingPipeline,
    EVAL_THRESHOLDS,
)


def test_model_instances_initialization():
    """Verify all 5 required model architectures are initialized."""
    models = get_model_instances()
    assert "logistic_regression" in models
    assert "decision_tree" in models
    assert "random_forest" in models
    assert any(gbm in models for gbm in ["xgboost", "lightgbm", "gradient_boosting"])
    assert "mlp_neural_network" in models


def test_evaluate_model_at_thresholds():
    """Verify threshold analysis metrics and high precision selection."""
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
    # Probabilities aligned with labels plus some noise
    y_prob = np.array([0.2, 0.7, 0.3, 0.65, 0.4, 0.8, 0.52, 0.3, 0.62, 0.58])

    results = evaluate_model_at_thresholds(y_true, y_prob, thresholds=EVAL_THRESHOLDS)

    assert "roc_auc" in results
    assert "pr_auc" in results
    assert "thresholds" in results
    assert len(results["thresholds"]) == len(EVAL_THRESHOLDS)

    for t in EVAL_THRESHOLDS:
        t_key = f"{t:.2f}"
        assert t_key in results["thresholds"]
        m = results["thresholds"][t_key]
        assert "precision" in m
        assert "recall" in m
        assert "f1_score" in m
        assert "buy_signals" in m

    # Check best_precision_threshold exists and is valid
    assert results["best_precision_threshold"] in results["thresholds"]


def test_pipeline_end_to_end(tmp_path: Path):
    """
    Creates synthetic processed partitions and runs end-to-end model training,
    threshold evaluation, artifact serialization, and metrics generation.
    """
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir(parents=True)

    # Generate synthetic training and validation datasets
    np.random.seed(42)
    n_train = 80
    n_val = 30
    n_features = 10

    feature_cols = [f"feat_{i}" for i in range(n_features)]
    
    # Train partition
    train_dates = pd.date_range("2020-01-01", periods=n_train, freq="B")
    X_train_df = pd.DataFrame(np.random.randn(n_train, n_features), columns=feature_cols)
    X_train_df.insert(0, "Ticker", "RELIANCE.NS")
    X_train_df.insert(0, "Date", train_dates)
    
    y_train_df = pd.DataFrame({
        "Date": train_dates,
        "Ticker": "RELIANCE.NS",
        "target_30d_return": np.random.normal(0.02, 0.05, n_train),
        "is_positive_growth": np.random.choice([0, 1], size=n_train, p=[0.45, 0.55]),
    })

    # Val partition
    val_dates = pd.date_range("2022-01-01", periods=n_val, freq="B")
    X_val_df = pd.DataFrame(np.random.randn(n_val, n_features), columns=feature_cols)
    X_val_df.insert(0, "Ticker", "RELIANCE.NS")
    X_val_df.insert(0, "Date", val_dates)

    y_val_df = pd.DataFrame({
        "Date": val_dates,
        "Ticker": "RELIANCE.NS",
        "target_30d_return": np.random.normal(0.01, 0.05, n_val),
        "is_positive_growth": np.random.choice([0, 1], size=n_val, p=[0.5, 0.5]),
    })

    # Save parquets
    X_train_df.to_parquet(processed_dir / "X_train.parquet", index=False)
    y_train_df.to_parquet(processed_dir / "y_train.parquet", index=False)
    X_val_df.to_parquet(processed_dir / "X_val.parquet", index=False)
    y_val_df.to_parquet(processed_dir / "y_val.parquet", index=False)

    pipeline = ModelTrainingPipeline(
        processed_dir=processed_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )

    # Test load_data metadata separation
    X_tr, y_tr, meta_tr, X_v, y_v, meta_v = pipeline.load_data()
    assert "Date" not in X_tr.columns and "Ticker" not in X_tr.columns
    assert "Date" not in X_v.columns and "Ticker" not in X_v.columns
    assert len(X_tr.columns) == n_features
    assert y_tr.ndim == 1 and len(y_tr) == n_train
    assert y_v.ndim == 1 and len(y_v) == n_val

    # Run training and evaluation
    compiled_metrics = pipeline.train_and_evaluate()

    # Check metrics report
    metrics_file = reports_dir / "metrics.json"
    assert metrics_file.exists()
    with open(metrics_file, "r", encoding="utf-8") as f:
        loaded_metrics = json.load(f)
    assert "logistic_regression" in loaded_metrics

    # Check validation predictions parquet
    pred_file = reports_dir / "validation_predictions.parquet"
    assert pred_file.exists()
    pred_df = pd.read_parquet(pred_file)
    assert len(pred_df) == n_val
    assert "Date" in pred_df.columns and "Ticker" in pred_df.columns
    assert "logistic_regression_prob" in pred_df.columns

    # Check model artifacts serialized to models_dir
    serialized_models = list(models_dir.glob("*.joblib"))
    assert len(serialized_models) >= 4  # All candidate models saved


def test_shap_explainer_dispatch():
    """Verify SHAP explainer dispatch helper handles import error or constructs explainer."""
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    sample_df = pd.DataFrame(np.random.rand(5, 3), columns=["f1", "f2", "f3"])

    try:
        import shap
        explainer = get_shap_explainer(model, sample_df)
        assert explainer is not None
    except ImportError:
        with pytest.raises(ImportError):
            get_shap_explainer(model, sample_df)
