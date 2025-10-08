import json, os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from joblib import dump
from src.preprocessing.tabular_preprocessing import build_tabular_preprocessor
from src.modeling.supervised_models import get_supervised_models
from src.modeling.unsupervised_models import get_unsupervised_models
from src.modeling.timeseries_models import get_timeseries_models, timeseries_cv_forecast
from src.evaluation.metrics_supervised import evaluate_supervised
from src.evaluation.metrics_unsupervised import evaluate_unsupervised
from src.evaluation.metrics_timeseries import evaluate_timeseries
from src.utils.logger import get_logger
from src.tracking.mlflow_utils import init_mlflow, start_run, log_params, log_metrics, log_artifacts

logger = get_logger(__name__)

def _cv(cfg, estimator, param_grid, X, y, scoring="neg_mean_squared_error"):
    cv_cfg = cfg.get("cv", {})
    n_splits = cv_cfg.get("n_splits", 5)
    shuffle = cv_cfg.get("shuffle", True)
    if scoring in ("f1_macro", "accuracy"):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=cfg.get("random_seed", 42))
    else:
        cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=cfg.get("random_seed", 42))
    strat = cfg.get("tuning", {}).get("strategy", "random")
    if strat == "grid":
        search = GridSearchCV(estimator, param_grid=param_grid or {}, cv=cv, scoring=scoring, n_jobs=cfg["tuning"].get("n_jobs", -1), refit=True)
    else:
        n_iter = cfg["tuning"].get("n_iter", 20)
        search = RandomizedSearchCV(estimator, param_distributions=param_grid or {}, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=cfg["tuning"].get("n_jobs", -1), random_state=cfg.get("random_seed", 42), refit=True)
    search.fit(X, y)
    return search

def run_supervised_experiment(df, cfg, problem_type="regression"):
    target = cfg["data"]["target_column"]
    pre, num, cat = build_tabular_preprocessor(df, target)
    models, grids = get_supervised_models(problem_type, pre)

    scoring = "f1_macro" if problem_type == "classification" else "neg_root_mean_squared_error"
    y = df[target].values
    X = df.drop(columns=[target])

    init_mlflow(cfg.get("experiment_name", "default_experiment"))
    results = []
    os.makedirs("reports", exist_ok=True)
    best_overall = None
    best_key = None
    for key, est in models.items():
        grid = grids.get(key, {})
        logger.info(f"Training {key} with scoring={scoring}")
        with start_run(run_name=f"supervised_{key}", tags={"problem_type": problem_type}):
            log_params(grid)
            search = _cv(cfg, est, grid, X, y, scoring=scoring)
            y_true, y_pred, metrics = evaluate_supervised(search.best_estimator_, X, y, problem_type)
            log_metrics(metrics)
            ts = int(time.time())
            tmp_path = f"reports/tmp_{key}_{ts}.json"
            with open(tmp_path, "w") as f:
                json.dump({"best_params": search.best_params_, "metrics": metrics}, f, indent=2)
            log_artifacts("reports")
            results.append({"model": key, "best_params": search.best_params_, "metrics": metrics})
            if (best_overall is None) or (metrics["score"] > best_overall["metrics"]["score"]):
                best_overall = {"model": key, "estimator": search.best_estimator_, "metrics": metrics}
                best_key = key

    ts = int(time.time())
    with open(f"reports/supervised_results_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)
    if best_overall:
        dump(best_overall["estimator"], f"reports/best_supervised_{best_key}_{ts}.joblib")
    logger.info("Supervised experiment finished.")

def run_unsupervised_experiment(df, cfg):
    init_mlflow(cfg.get("experiment_name", "default_experiment"))
    models = get_unsupervised_models()
    X = df.select_dtypes(include=["number"]).copy()
    results = []
    for key, est in models.items():
        with start_run(run_name=f"unsupervised_{key}"):
            est.fit(X)
            metrics = evaluate_unsupervised(est, X)
            log_metrics(metrics)
            results.append({"model": key, "metrics": metrics})
    ts = int(time.time())
    with open(f"reports/unsupervised_results_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Unsupervised experiment finished.")

def run_timeseries_experiment(df, cfg):
    dt_col = cfg["data"]["datetime_column"]
    target_col = cfg["data"]["target_column"]
    ts_cfg = cfg.get("timeseries", {})
    init_mlflow(cfg.get("experiment_name", "default_experiment"))
    models = get_timeseries_models(dt_col, target_col, ts_cfg)

    results = []
    for key, model in models.items():
        with start_run(run_name=f"timeseries_{key}", tags={"type": "forecast"}):
            y_true, y_pred = timeseries_cv_forecast(df[[dt_col, target_col]].copy(), dt_col, target_col, model, n_splits=cfg.get("cv", {}).get("n_splits", 5))
            from src.evaluation.metrics_timeseries import evaluate_timeseries
            metrics = evaluate_timeseries(y_true, y_pred)
            log_metrics(metrics)
            results.append({"model": key, "metrics": metrics})
    ts = int(time.time())
    with open(f"reports/timeseries_results_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Time series experiment finished.")
