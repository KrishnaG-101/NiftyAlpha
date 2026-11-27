"""
NiftyAlpha - Machine Learning & Modeling Module (Phase 4)
=========================================================
Production-grade machine learning model training, precision-targeted threshold
evaluation, XAI (SHAP) explainer abstraction, and artifact serialization.

Architectures:
1. Baseline: Logistic Regression (L2 penalty, max_iter=1000)
2. Decision Tree: DecisionTreeClassifier (max_depth=5)
3. Random Forest: RandomForestClassifier (n_estimators=100, max_depth=10)
4. Gradient Boosting: XGBClassifier (or LGBMClassifier fallback)
5. Neural Network: MLPClassifier (hidden_layer_sizes=(64, 32), early_stopping=True)

Precision-Driven Thresholding:
- Evaluates ROC-AUC, PR-AUC across validation set.
- Threshold analysis across [0.50, 0.55, 0.60, 0.65].
- Identifies optimal high-precision threshold (>60% precision with non-zero recall)
  to minimize false positives in quantitative trade execution.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NiftyAlpha.Models")

# Project Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

EVAL_THRESHOLDS = [0.50, 0.55, 0.60, 0.65]


# ==============================================================================
# Model Factory
# ==============================================================================

def get_model_instances() -> Dict[str, Any]:
    """
    Instantiates and returns the configured candidate model suite.
    """
    models: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            C=0.1,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        ),
    }

    # Gradient Boosting: XGBoost preferred, LightGBM fallback
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
    except ImportError:
        try:
            from lightgbm import LGBMClassifier
            models["lightgbm"] = LGBMClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.03,
                random_state=42,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            models["gradient_boosting"] = GradientBoostingClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.03,
                random_state=42,
            )

    # Neural Network MLP
    models["mlp_neural_network"] = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        early_stopping=True,
        max_iter=500,
        random_state=42,
    )

    return models


# ==============================================================================
# Hyperparameter Search Spaces & Chronological TimeSeriesSplit Tuning
# ==============================================================================

PARAM_SEARCH_SPACES: Dict[str, Dict[str, List[Any]]] = {
    "xgboost": {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
        "n_estimators": [80, 100, 120, 150],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    },
    "random_forest": {
        "n_estimators": [80, 100, 150, 200],
        "max_depth": [6, 8, 10, 12, None],
        "min_samples_leaf": [10, 20, 30, 50],
        "max_features": ["sqrt", "log2", 0.7, 0.8],
    },
    "decision_tree": {
        "max_depth": [3, 4, 5, 6, 8],
        "min_samples_leaf": [10, 20, 30, 50, 100],
        "criterion": ["gini", "entropy"],
    },
    "logistic_regression": {
        "C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    },
}


def tune_hyperparameters(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_splits: int = 5,
    n_iter: int = 8,
    scoring: str = "average_precision",
    search_type: str = "randomized",
    param_grid: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    n_jobs: int = 1,
) -> Tuple[Any, Dict[str, Any], float, pd.DataFrame]:
    """
    Tunes hyperparameters for a specified model architecture using chronological
    TimeSeriesSplit to strictly prevent temporal lookahead leakage.

    Parameters:
    -----------
    model_name : str
        Model identifier ('xgboost', 'random_forest', 'decision_tree', 'logistic_regression').
    X_train : pd.DataFrame
        Training feature matrix without Date/Ticker columns.
    y_train : np.ndarray
        1D binary target series.
    n_splits : int
        Number of forward chronological splits for TimeSeriesSplit (default: 5).
    n_iter : int
        Number of parameter combinations to sample for RandomizedSearchCV (default: 8).
    scoring : str
        Scikit-learn scoring string, e.g. 'average_precision' (PR-AUC) or 'roc_auc'.
    search_type : str
        'randomized' for RandomizedSearchCV, or 'grid' for GridSearchCV.
    param_grid : Optional[Dict[str, Any]]
        Custom parameter search dictionary. If None, uses defaults from PARAM_SEARCH_SPACES.
    random_state : int
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel worker processes (default: 1).

    Returns:
    --------
    best_estimator : Any
        Tuned model fitted on the entire training dataset.
    best_params : Dict[str, Any]
        Dictionary of optimal hyperparameters found.
    best_score : float
        Mean cross-validation score across the TimeSeriesSplit folds.
    cv_results_df : pd.DataFrame
        Tabular summary of evaluated parameter combinations and their fold performance.
    """
    models = get_model_instances()
    canonical_name = model_name.lower().strip()

    # Handle alias matching (e.g. 'xgb' -> 'xgboost', 'rf' -> 'random_forest', 'dt' -> 'decision_tree')
    alias_map = {
        "xgb": "xgboost",
        "rf": "random_forest",
        "dt": "decision_tree",
        "lr": "logistic_regression",
        "logistic": "logistic_regression",
    }
    canonical_name = alias_map.get(canonical_name, canonical_name)

    if canonical_name not in models:
        # Fallback to key containing name
        matched_keys = [k for k in models if canonical_name in k]
        if matched_keys:
            canonical_name = matched_keys[0]
        else:
            raise ValueError(
                f"Unknown model_name '{model_name}'. Available: {list(models.keys())}"
            )

    base_model = models[canonical_name]

    if param_grid is None:
        if canonical_name not in PARAM_SEARCH_SPACES:
            raise ValueError(
                f"No default parameter search space defined for '{canonical_name}'. "
                f"Please pass param_grid explicitly."
            )
        search_space = PARAM_SEARCH_SPACES[canonical_name]
    else:
        search_space = param_grid

    logger.info(
        "Initializing chronological TimeSeriesSplit(n_splits=%d) for '%s'...",
        n_splits,
        canonical_name,
    )
    tscv = TimeSeriesSplit(n_splits=n_splits)

    if search_type.lower() == "grid":
        logger.info(
            "Executing GridSearchCV across %d parameter keys...",
            len(search_space),
        )
        search = GridSearchCV(
            estimator=base_model,
            param_grid=search_space,
            cv=tscv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
        )
    else:
        logger.info(
            "Executing RandomizedSearchCV (n_iter=%d) across parameter distributions...",
            n_iter,
        )
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=search_space,
            n_iter=n_iter,
            cv=tscv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=n_jobs,
            refit=True,
        )

    search.fit(X_train, y_train)

    best_estimator = search.best_estimator_
    best_params = search.best_params_
    best_score = float(search.best_score_)
    cv_results_df = pd.DataFrame(search.cv_results_)

    logger.info(
        "[%s Tuning Complete] Best %s Score: %.4f | Optimal Params: %s",
        canonical_name.upper(),
        scoring,
        best_score,
        best_params,
    )

    return best_estimator, best_params, best_score, cv_results_df


# ==============================================================================
# SHAP Explainer Helper
# ==============================================================================

def get_shap_explainer(model: Any, X_sample: pd.DataFrame) -> Any:
    """
    Returns the appropriate SHAP explainer based on model architecture:
    - TreeExplainer for Tree-based models (Random Forest, Decision Tree, XGBoost, LightGBM)
    - LinearExplainer for Logistic Regression
    - KernelExplainer for Neural Networks (MLPClassifier) and generic models.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP library is not installed. Please run 'pip install shap'.")
        raise ImportError(
            "SHAP is required for model explainability. Install it via 'pip install shap'."
        )

    model_name = model.__class__.__name__.lower()

    if any(tree_kw in model_name for tree_kw in ["tree", "forest", "xgb", "lgbm", "gradientboost"]):
        return shap.TreeExplainer(model)
    elif "logistic" in model_name or "linear" in model_name:
        return shap.LinearExplainer(model, X_sample)
    else:
        # Default / MLP / Neural Network
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        background = (
            shap.sample(X_sample, min(50, len(X_sample))) if len(X_sample) > 50 else X_sample
        )
        return shap.KernelExplainer(predict_fn, background)


# ==============================================================================
# Evaluation & Thresholding
# ==============================================================================

def evaluate_model_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = EVAL_THRESHOLDS,
) -> Dict[str, Any]:
    """
    Evaluates classification metrics across probability thresholds, focusing
    on Class 1 (Buy Signals) precision.
    """
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    threshold_results: Dict[str, Dict[str, Any]] = {}
    best_threshold: Optional[str] = None
    best_prec = 0.0

    for t in thresholds:
        t_key = f"{t:.2f}"
        y_pred = (y_prob >= t).astype(int)
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        buy_signals = int(np.sum(y_pred))

        threshold_results[t_key] = {
            "threshold": t,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "buy_signals": buy_signals,
            "total_samples": len(y_true),
        }

        # Check precision-driven criterion: >= 60% precision with non-zero recall
        if prec >= 0.60 and rec > 0.0:
            if prec > best_prec:
                best_prec = prec
                best_threshold = t_key

    # Fallback to highest precision threshold with non-zero recall if >= 60% not reached
    if best_threshold is None:
        valid_candidates = [
            (k, v) for k, v in threshold_results.items() if v["recall"] > 0.0
        ]
        if valid_candidates:
            best_threshold = max(valid_candidates, key=lambda x: x[1]["precision"])[0]
        else:
            best_threshold = f"{thresholds[0]:.2f}"

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "thresholds": threshold_results,
        "best_precision_threshold": best_threshold,
        "best_threshold_metrics": threshold_results[best_threshold],
    }


# ==============================================================================
# Model Pipeline Orchestrator
# ==============================================================================

class ModelTrainingPipeline:
    """
    Coordinates data loading, model fitting, precision threshold evaluation,
    and artifact serialization.
    """

    def __init__(
        self,
        processed_dir: Path = DEFAULT_PROCESSED_DIR,
        models_dir: Path = DEFAULT_MODELS_DIR,
        reports_dir: Path = DEFAULT_REPORTS_DIR,
        tune: bool = False,
        tune_model: str = "xgboost",
        n_splits: int = 5,
        n_iter: int = 8,
    ):
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.tune = tune
        self.tune_model = tune_model.lower().strip()
        self.n_splits = n_splits
        self.n_iter = n_iter

    def load_data(
        self,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        Loads processed parquet partitions, strictly separating Date and Ticker
        metadata from feature matrices and extracting the 1D binary target series.
        """
        x_train_path = self.processed_dir / "X_train.parquet"
        y_train_path = self.processed_dir / "y_train.parquet"
        x_val_path = self.processed_dir / "X_val.parquet"
        y_val_path = self.processed_dir / "y_val.parquet"

        for p in [x_train_path, y_train_path, x_val_path, y_val_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Required dataset partition missing at {p}. "
                    f"Please run 'make features' or 'python src/features.py' first."
                )

        logger.info("Loading processed partitions from %s...", self.processed_dir)
        X_train_raw = pd.read_parquet(x_train_path)
        y_train_raw = pd.read_parquet(y_train_path)
        X_val_raw = pd.read_parquet(x_val_path)
        y_val_raw = pd.read_parquet(y_val_path)

        meta_cols = ["Date", "Ticker"]
        meta_train = X_train_raw[meta_cols].copy()
        meta_val = X_val_raw[meta_cols].copy()

        # Pure feature matrices (excluding Date and Ticker)
        feature_cols = [c for c in X_train_raw.columns if c not in meta_cols]
        X_train = X_train_raw[feature_cols].copy()
        X_val = X_val_raw[feature_cols].copy()

        # 1D Target series: Prefer target_alpha_outperform for excess return modeling, fallback to is_positive_growth
        target_col = (
            "target_alpha_outperform"
            if "target_alpha_outperform" in y_train_raw.columns
            else "is_positive_growth"
        )
        y_train = y_train_raw[target_col].astype(int).values
        y_val = y_val_raw[target_col].astype(int).values
        logger.info("Using target column '%s' for model optimization.", target_col)

        logger.info(
            "Data loaded successfully. Features: %d | Train: %d rows | Val: %d rows",
            len(feature_cols),
            len(X_train),
            len(X_val),
        )
        return X_train, y_train, meta_train, X_val, y_val, meta_val

    def train_and_evaluate(self) -> Dict[str, Any]:
        """
        Trains all candidate models, computes evaluation metrics with threshold analysis,
        and saves artifacts and reports.
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        X_train, y_train, meta_train, X_val, y_val, meta_val = self.load_data()
        models = get_model_instances()

        compiled_metrics: Dict[str, Any] = {}
        val_predictions: Dict[str, np.ndarray] = {
            "Date": meta_val["Date"].values,
            "Ticker": meta_val["Ticker"].values,
            "y_true": y_val,
        }

        # Automated hyperparameter tuning if requested
        if self.tune:
            logger.info(
                "Automated hyperparameter tuning requested for '%s' using TimeSeriesSplit (n_splits=%d, n_iter=%d)...",
                self.tune_model,
                self.n_splits,
                self.n_iter,
            )
            try:
                tuned_model, best_params, best_score, cv_results = tune_hyperparameters(
                    model_name=self.tune_model,
                    X_train=X_train,
                    y_train=y_train,
                    n_splits=self.n_splits,
                    n_iter=self.n_iter,
                    scoring="average_precision",
                )
                alias_map = {
                    "xgb": "xgboost",
                    "rf": "random_forest",
                    "dt": "decision_tree",
                    "lr": "logistic_regression",
                }
                c_name = alias_map.get(self.tune_model, self.tune_model)
                target_key = None
                if c_name in models:
                    target_key = c_name
                else:
                    for k in models:
                        if c_name in k:
                            target_key = k
                            break

                if target_key:
                    models[target_key] = tuned_model
                    clean_best_params = {}
                    for k, v in best_params.items():
                        if isinstance(v, (np.integer, np.floating)):
                            clean_best_params[k] = v.item()
                        elif v is None or isinstance(v, (int, float, str, bool)):
                            clean_best_params[k] = v
                        else:
                            clean_best_params[k] = str(v)

                    tuning_summary = {
                        "model": target_key,
                        "cv_method": f"TimeSeriesSplit(n_splits={self.n_splits})",
                        "scoring": "average_precision",
                        "best_score": best_score,
                        "best_params": clean_best_params,
                        "candidates_evaluated": len(cv_results),
                    }
                    compiled_metrics["_hyperparameter_tuning"] = tuning_summary
                    tuning_path = self.reports_dir / "hyperparameter_tuning.json"
                    with open(tuning_path, "w", encoding="utf-8") as f:
                        json.dump(tuning_summary, f, indent=2)
                    logger.info("Saved hyperparameter tuning results to %s", tuning_path)
            except Exception as e:
                logger.error(
                    "Hyperparameter tuning encountered error: %s. Proceeding with baseline models.",
                    e,
                )

        logger.info("Starting model training and evaluation suite...")

        for name, model in models.items():
            logger.info("--> Training %s...", name.upper())
            model.fit(X_train, y_train)

            # Generate validation prediction probabilities for positive class (Buy)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_val)[:, 1]
            else:
                y_prob = model.predict(X_val)

            val_predictions[f"{name}_prob"] = y_prob

            # Compute precision threshold metrics
            eval_metrics = evaluate_model_at_thresholds(y_val, y_prob)
            compiled_metrics[name] = eval_metrics

            best_t = eval_metrics["best_precision_threshold"]
            best_m = eval_metrics["best_threshold_metrics"]

            logger.info(
                "[%s] ROC-AUC: %.4f | PR-AUC: %.4f | Best Thresh (%s) -> Prec: %.4f, Recall: %.4f (Signals: %d)",
                name,
                eval_metrics["roc_auc"],
                eval_metrics["pr_auc"],
                best_t,
                best_m["precision"],
                best_m["recall"],
                best_m["buy_signals"],
            )

            # Serialize model artifact
            model_path = self.models_dir / f"{name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info("Serialized model artifact to %s", model_path)

        # Save compiled metrics to reports/metrics.json
        metrics_path = self.reports_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(compiled_metrics, f, indent=2)
        logger.info("Compiled evaluation metrics saved to %s", metrics_path)

        # Save validation predictions for backtesting & dashboard consumption
        val_pred_df = pd.DataFrame(val_predictions)
        val_pred_path = self.reports_dir / "validation_predictions.parquet"
        val_pred_df.to_parquet(val_pred_path, index=False, engine="pyarrow")
        logger.info("Validation predictions persisted to %s", val_pred_path)

        return compiled_metrics


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NiftyAlpha Machine Learning & Modeling Module (Phase 4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR),
        help="Directory containing processed parquet partitions.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help="Directory where trained model artifacts will be saved.",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory where evaluation metrics and predictions will be stored.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run automated hyperparameter tuning with TimeSeriesSplit before evaluation.",
    )
    parser.add_argument(
        "--tune-model",
        type=str,
        default="xgboost",
        help="Model architecture to tune (e.g. xgboost, random_forest, decision_tree, logistic_regression).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of chronological splits for TimeSeriesSplit cross-validation.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=8,
        help="Number of parameter search iterations for RandomizedSearchCV.",
    )

    args = parser.parse_args()

    pipeline = ModelTrainingPipeline(
        processed_dir=Path(args.processed_dir),
        models_dir=Path(args.models_dir),
        reports_dir=Path(args.reports_dir),
        tune=args.tune,
        tune_model=args.tune_model,
        n_splits=args.n_splits,
        n_iter=args.n_iter,
    )
    pipeline.train_and_evaluate()


if __name__ == "__main__":
    main()
